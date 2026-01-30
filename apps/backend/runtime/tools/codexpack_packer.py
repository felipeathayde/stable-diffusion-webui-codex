"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Offline CodexPack generator (GGUF → `*.codexpack.gguf`) for packed CUDA execution.
Repacks GGML `Q4_K` 2D weights into the CodexPack `cuda.ggml_q4_k.linear.tilepack_v1` layout and emits an auto-detectable GGUF.
Non tile-aligned `Q4_K` weights are dequantized offline to `F16` (avoids per-forward dequantization and dequant-upfront VRAM blowups).

Symbols (top-level; keep in sync; no ghosts):
- `CodexPackPackError` (class): Raised when packing fails (unsupported tensor types/shapes, invalid inputs, IO issues).
- `Q4K_BLOCK_BYTES` (constant): GGML Q4_K block byte size (derived from `GGML_QUANT_SIZES`).
- `pack_gguf_to_codexpack_v1` (function): Converts a base GGUF into a CodexPack GGUF (packed tile-aligned Q4_K linears + copied float tensors + FP16 fallback).
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from apps.backend.quantization.codexpack_keymaps import is_supported_codexpack_keymap_id
from apps.backend.quantization.dequant import dequantize_blocks_Q4_K
from apps.backend.quantization.gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from apps.backend.quantization.gguf.constants import GGUFValueType
from apps.backend.quantization.gguf.constants import GGML_QUANT_SIZES
from apps.backend.quantization.gguf.codexpack import (
    CODEXPACK_SCHEMA,
    CODEXPACK_SCHEMA_VERSION,
    KERNEL_ID_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1,
    TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1,
    TILE_M_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1,
    is_codexpack_gguf,
    load_codexpack_manifest_v1,
)


class CodexPackPackError(RuntimeError):
    pass


_FLOATLIKE_GGML_TYPES = {
    GGMLQuantizationType.F16,
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F64,
    GGMLQuantizationType.I8,
    GGMLQuantizationType.I16,
    GGMLQuantizationType.I32,
    GGMLQuantizationType.I64,
}

Q4K_BLOCK_BYTES = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K][1]


def _read_scalar_metadata_value(field) -> Any:
    vtype = field.types[0]
    if vtype == GGUFValueType.STRING:
        raw = field.parts[field.data[0]]
        return bytes(raw).decode("utf-8")
    raw = field.parts[field.data[0]]
    if not isinstance(raw, np.ndarray) or raw.size != 1:
        raise CodexPackPackError(f"Expected scalar metadata field {field.name!r}, got: {type(raw)!r} shape={getattr(raw, 'shape', None)}")
    return raw.reshape(()).item()


def _copy_scalar_metadata(*, src: GGUFReader, dst: GGUFWriter) -> None:
    for key, field in src.fields.items():
        if key.startswith("GGUF."):
            continue
        if key.startswith("codex.pack."):
            continue
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            raise CodexPackPackError(
                "Source GGUF contains ARRAY metadata which this CodexPack packer does not copy (v1). "
                f"key={key!r}. Convert the source with Codex GGUF converter tooling or add explicit support."
            )
        dst.add_key_value(key, _read_scalar_metadata_value(field), vtype)


def _dequantize_q4k_matrix_to_fp16(raw_bytes: np.ndarray, *, out_features: int, in_features: int) -> np.ndarray:
    if raw_bytes.dtype != np.uint8:
        raw_bytes = raw_bytes.view(np.uint8)

    tile_k = TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1
    if in_features % tile_k != 0:
        raise CodexPackPackError(f"in_features must be multiple of {tile_k}; got {in_features}")

    k_tiles = in_features // tile_k
    bytes_per_row = k_tiles * Q4K_BLOCK_BYTES
    if raw_bytes.shape != (out_features, bytes_per_row):
        raise CodexPackPackError(
            "Q4_K byte tensor has unexpected shape. "
            f"expected {(out_features, bytes_per_row)}, got {tuple(int(v) for v in raw_bytes.shape)}"
        )

    blocks_np = raw_bytes.reshape((out_features * k_tiles, Q4K_BLOCK_BYTES))
    if not blocks_np.flags.writeable:
        blocks_np = np.array(blocks_np, copy=True)

    blocks = torch.from_numpy(blocks_np).to(dtype=torch.uint8)
    out = torch.empty((blocks.shape[0], tile_k), dtype=torch.float16)

    chunk_blocks = 8192
    for start in range(0, blocks.shape[0], chunk_blocks):
        end = min(start + chunk_blocks, blocks.shape[0])
        out[start:end] = dequantize_blocks_Q4_K(
            blocks[start:end],
            TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1,
            Q4K_BLOCK_BYTES,
            dtype=torch.float16,
        )

    mat = out.reshape((out_features, k_tiles, tile_k)).reshape((out_features, in_features))
    return np.ascontiguousarray(mat.cpu().numpy())


def _pack_q4k_tilepack_v1(raw_bytes: np.ndarray, *, out_features: int, in_features: int) -> np.ndarray:
    if raw_bytes.dtype != np.uint8:
        raw_bytes = raw_bytes.view(np.uint8)

    tile_m = TILE_M_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1
    tile_k = TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1
    if out_features % tile_m != 0:
        raise CodexPackPackError(f"out_features must be multiple of {tile_m}; got {out_features}")
    if in_features % tile_k != 0:
        raise CodexPackPackError(f"in_features must be multiple of {tile_k}; got {in_features}")

    k_tiles = in_features // tile_k
    bytes_per_row = k_tiles * Q4K_BLOCK_BYTES
    if raw_bytes.shape != (out_features, bytes_per_row):
        raise CodexPackPackError(
            "Q4_K byte tensor has unexpected shape. "
            f"expected {(out_features, bytes_per_row)}, got {tuple(int(v) for v in raw_bytes.shape)}"
        )

    # Base GGUF layout is row-major (out-major): [out][k_tile][144 bytes].
    blocks = raw_bytes.reshape((out_features, k_tiles, Q4K_BLOCK_BYTES))

    # CodexPack layout is tile-major: [m_tile][k_tile][m_lane][144 bytes].
    m_tiles = out_features // tile_m
    packed_u8 = (
        blocks.reshape((m_tiles, tile_m, k_tiles, Q4K_BLOCK_BYTES))
        .transpose(0, 2, 1, 3)
        .reshape((-1,))
    )
    return np.ascontiguousarray(packed_u8.view(np.int8))


def _compute_dora_norm_out_q4k(raw_bytes: np.ndarray, *, out_features: int, in_features: int) -> np.ndarray:
    if raw_bytes.dtype != np.uint8:
        raw_bytes = raw_bytes.view(np.uint8)
    tile_k = TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1
    if in_features % tile_k != 0:
        raise CodexPackPackError(f"in_features must be multiple of {tile_k}; got {in_features}")

    k_tiles = in_features // tile_k
    blocks_per_row = k_tiles
    bytes_per_row = blocks_per_row * Q4K_BLOCK_BYTES
    if raw_bytes.shape != (out_features, bytes_per_row):
        raise CodexPackPackError(
            "Q4_K byte tensor has unexpected shape. "
            f"expected {(out_features, bytes_per_row)}, got {tuple(int(v) for v in raw_bytes.shape)}"
        )

    # Flatten to (n_blocks, 144) row-major.
    blocks_np = raw_bytes.reshape((out_features * blocks_per_row, Q4K_BLOCK_BYTES))
    if not blocks_np.flags.writeable:
        blocks_np = np.array(blocks_np, copy=True)

    blocks = torch.from_numpy(blocks_np).to(dtype=torch.uint8)
    acc = torch.zeros((out_features,), dtype=torch.float64)

    # Chunk to avoid materializing the full dequant matrix.
    chunk_blocks = 8192
    for start in range(0, blocks.shape[0], chunk_blocks):
        end = min(start + chunk_blocks, blocks.shape[0])
        chunk = blocks[start:end]
        w = dequantize_blocks_Q4_K(
            chunk,
            TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1,
            Q4K_BLOCK_BYTES,
            dtype=torch.float32,
        )
        sums = (w * w).sum(dim=1, dtype=torch.float64)
        row_idx = torch.arange(start, end, dtype=torch.int64) // blocks_per_row
        acc.index_add_(0, row_idx, sums)

    norms = torch.sqrt(acc).to(dtype=torch.float32).cpu().numpy()
    return np.ascontiguousarray(norms.astype(np.float32, copy=False))


def pack_gguf_to_codexpack_v1(
    src_gguf_path: str,
    dst_codexpack_gguf_path: str,
    *,
    keymap_id: str,
    cuda_sm_min: int = 86,
    internal_prefix: str = "__codexpack__.",
) -> None:
    """Create a CodexPack GGUF from a base GGUF (offline).

    v1 policy:
    - Packs every 2D tile-aligned `Q4_K` tensor (assumed Linear weight) and drops the raw quant tensor from the output.
    - For non tile-aligned 2D `Q4_K` tensors, dequantizes offline to `F16` and stores them as normal float tensors.
    - Copies float-like tensors unchanged.
    - Fails loud on any non-`Q4_K` quant tensors or unsupported metadata types.
    """
    if not is_supported_codexpack_keymap_id(keymap_id):
        raise CodexPackPackError(f"keymap_id is unknown to this build: {keymap_id!r}")
    if not internal_prefix.startswith("__codexpack__."):
        raise CodexPackPackError(f"internal_prefix must start with '__codexpack__.'; got: {internal_prefix!r}")
    if int(cuda_sm_min) != 86:
        raise CodexPackPackError(f"CodexPack v1 targets SM86 only; got cuda_sm_min={cuda_sm_min!r}")

    src = GGUFReader(src_gguf_path)
    if is_codexpack_gguf(src):
        pack = load_codexpack_manifest_v1(src)
        raise CodexPackPackError(
            "Refusing to pack an existing CodexPack GGUF. "
            f"schema_version={pack.schema_version} keymap_id={pack.keymap_id!r} kernel_id={pack.kernel_id!r}."
        )

    out_path = Path(dst_codexpack_gguf_path)
    if out_path.exists():
        raise CodexPackPackError(f"Output already exists: {str(out_path)!r}")

    manifest_entries: list[dict[str, Any]] = []
    float_keys: list[str] = []
    fallback_fp16_keys: list[str] = []

    w = GGUFWriter(str(out_path), arch="codexpack")
    # `GGUFWriter` writes `general.architecture` eagerly. Codex outputs use a custom metadata schema,
    # so remove it and let source metadata drive what keys are present.
    try:
        for shard in w.kv_data:
            if isinstance(shard, dict):
                shard.pop("general.architecture", None)
    except Exception:
        pass
    _copy_scalar_metadata(src=src, dst=w)

    for t in src.tensors:
        name = t.name
        ggml_type = t.tensor_type
        if name.startswith(internal_prefix):
            raise CodexPackPackError(f"Source GGUF already contains reserved internal prefix tensor: {name!r}")

        # Float-like tensors are copied unchanged.
        if ggml_type in _FLOATLIKE_GGML_TYPES:
            float_keys.append(name)
            w.add_tensor(name, np.array(t.data, copy=True))
            continue

        # Quant tensors: v1 only supports Q4_K and requires 2D tile-aligned shapes.
        if ggml_type != GGMLQuantizationType.Q4_K:
            raise CodexPackPackError(
                "CodexPack v1 only supports Q4_K quant tensors. "
                f"found {ggml_type.name} tensor {name!r}."
            )

        real_shape = tuple(int(v) for v in reversed(t.shape.tolist()))
        if len(real_shape) != 2:
            raise CodexPackPackError(f"CodexPack v1 only supports 2D Q4_K tensors; got {name!r} shape={real_shape}")
        out_features, in_features = int(real_shape[0]), int(real_shape[1])

        raw_bytes = t.data.view(np.uint8)

        if out_features % TILE_M_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1 != 0 or in_features % TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1 != 0:
            # Fallback: keep the tensor, but store as FP16 (the packed kernel requires tile alignment).
            fallback_fp16_keys.append(name)
            float_keys.append(name)
            w.add_tensor(name, _dequantize_q4k_matrix_to_fp16(raw_bytes, out_features=out_features, in_features=in_features))
            continue

        packed = _pack_q4k_tilepack_v1(raw_bytes, out_features=out_features, in_features=in_features)
        dora_norm_out = _compute_dora_norm_out_q4k(raw_bytes, out_features=out_features, in_features=in_features)

        packed_name = f"{internal_prefix}{name}.packed"
        dora_name = f"{internal_prefix}{name}.dora_norm_out"

        w.add_tensor(packed_name, packed)
        w.add_tensor(dora_name, dora_norm_out)

        manifest_entries.append(
            {
                "kind": "linear_weight",
                "param_key": name,
                "shape": [out_features, in_features],
                "qtype": "Q4_K",
                "tensors": {"packed": packed_name, "dora_norm_out": dora_name},
            }
        )

    manifest = {
        "schema_version": CODEXPACK_SCHEMA_VERSION,
        "keymap_id": keymap_id,
        "kernel_id": KERNEL_ID_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1,
        "targets": {"backend": "cuda", "cuda_sm_min": int(cuda_sm_min)},
        "entries": manifest_entries,
        "float_keys": float_keys,
        "fallback_fp16_keys": fallback_fp16_keys,
    }

    w.add_key_value("codex.pack.schema", CODEXPACK_SCHEMA, GGUFValueType.STRING)
    w.add_key_value("codex.pack.schema_version", CODEXPACK_SCHEMA_VERSION, GGUFValueType.UINT32)
    w.add_key_value("codex.pack.keymap_id", keymap_id, GGUFValueType.STRING)
    w.add_key_value("codex.pack.kernel_id", KERNEL_ID_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1, GGUFValueType.STRING)
    w.add_key_value("codex.pack.cuda_sm_min", int(cuda_sm_min), GGUFValueType.UINT32)
    w.add_key_value("codex.pack.manifest_json", json.dumps(manifest, separators=(",", ":"), sort_keys=True), GGUFValueType.STRING)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()

    # Validate output deterministically (fail loud).
    out_reader = GGUFReader(str(out_path))
    load_codexpack_manifest_v1(out_reader)


def _build_cli(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pack a base GGUF into a CodexPack GGUF (tilepack_v1).")
    p.add_argument("--in", dest="src", required=True, help="Input base GGUF path (Q4_K mixed allowed).")
    p.add_argument("--out", dest="dst", required=True, help="Output CodexPack GGUF path (will be created).")
    p.add_argument(
        "--keymap-id",
        dest="keymap_id",
        required=True,
        help="CodexPack keymap id (must be supported by this build).",
    )
    return p.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    ns = _build_cli(argv)
    pack_gguf_to_codexpack_v1(ns.src, ns.dst, keymap_id=ns.keymap_id)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "CodexPackPackError",
    "pack_gguf_to_codexpack_v1",
]
