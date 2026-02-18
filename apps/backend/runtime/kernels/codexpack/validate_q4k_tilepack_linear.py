"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: CUDA correctness harness for CodexPack packed Q4_K linear (tilepack_v1).
Validates `torch.ops.codexpack.q4k_tilepack_linear(...)` against a baseline:
Q4_K bytes → dequantize (fp16) → `torch.nn.functional.linear`.
CLI success output is emitted via shared infra stdout helpers.

Symbols (top-level; keep in sync; no ghosts):
- `CodexPackCudaValidationError` (exception): Raised when validation prerequisites are not met.
- `validate_q4k_tilepack_linear` (function): Runs one correctness check and returns max abs error.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from apps.backend.infra.stdio import write_stdout
from apps.backend.quantization.api import quantize_numpy
from apps.backend.quantization.dequant import dequantize_blocks_Q4_K
from apps.backend.quantization.gguf.codexpack import (
    Q4K_BLOCK_BYTES,
    TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1,
    TILE_M_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1,
)
from apps.backend.quantization.gguf.constants import GGMLQuantizationType
from apps.backend.runtime.ops import codexpack_cuda


class CodexPackCudaValidationError(RuntimeError):
    pass


def _pack_q4k_tilepack_v1(raw_bytes_u8: np.ndarray, *, out_features: int, in_features: int) -> np.ndarray:
    if raw_bytes_u8.dtype != np.uint8:
        raw_bytes_u8 = raw_bytes_u8.view(np.uint8)

    tile_m = TILE_M_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1
    tile_k = TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1
    block_bytes = Q4K_BLOCK_BYTES

    if out_features % tile_m != 0:
        raise CodexPackCudaValidationError(f"out_features must be multiple of {tile_m}; got {out_features}")
    if in_features % tile_k != 0:
        raise CodexPackCudaValidationError(f"in_features must be multiple of {tile_k}; got {in_features}")

    k_tiles = in_features // tile_k
    bytes_per_row = k_tiles * block_bytes
    if raw_bytes_u8.shape != (out_features, bytes_per_row):
        raise CodexPackCudaValidationError(
            "Q4_K byte tensor has unexpected shape. "
            f"expected {(out_features, bytes_per_row)}, got {tuple(int(v) for v in raw_bytes_u8.shape)}"
        )

    blocks = raw_bytes_u8.reshape((out_features, k_tiles, block_bytes))
    m_tiles = out_features // tile_m
    packed_u8 = (
        blocks.reshape((m_tiles, tile_m, k_tiles, block_bytes))
        .transpose(0, 2, 1, 3)
        .reshape((-1,))
    )
    return np.ascontiguousarray(packed_u8.view(np.int8))


def validate_q4k_tilepack_linear(
    *,
    out_features: int,
    in_features: int,
    batch: int,
    seed: int,
    bias: bool,
    rtol: float,
    atol: float,
) -> float:
    if not torch.cuda.is_available():
        raise CodexPackCudaValidationError("CUDA is not available (torch.cuda.is_available() == False).")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 6):
        raise CodexPackCudaValidationError(
            "CodexPack packed-kernel execution requires NVIDIA SM86+ (RTX 30xx baseline). "
            f"got compute_capability={(major, minor)}."
        )

    if not codexpack_cuda.available():
        last = codexpack_cuda.last_error()
        raise CodexPackCudaValidationError(
            "codexpack_cuda extension is not available.\n"
            f"{(f'Last error:\\n{last}\\n' if last else '')}"
            "Build in-place:\n"
            "  CODEX_ROOT=\"$(git rev-parse --show-toplevel)\"\n"
            "  cd \"$CODEX_ROOT/apps/backend/runtime/kernels/codexpack\"\n"
            "  PYTHONPATH=\"$CODEX_ROOT\" \"$CODEX_ROOT/.venv/bin/python\" setup.py build_ext --inplace\n"
        )
    if not hasattr(torch.ops, "codexpack") or not hasattr(torch.ops.codexpack, "q4k_tilepack_linear"):
        raise CodexPackCudaValidationError(
            "codexpack_cuda extension imported, but `torch.ops.codexpack.q4k_tilepack_linear` is missing."
        )

    rng = np.random.RandomState(int(seed))
    weight_f32 = rng.randn(int(out_features), int(in_features)).astype(np.float32, copy=False)
    q4k_bytes = quantize_numpy(weight_f32, GGMLQuantizationType.Q4_K)  # (out_features, k_tiles*144) uint8
    raw_u8 = np.ascontiguousarray(q4k_bytes.view(np.uint8))
    packed_i8 = _pack_q4k_tilepack_v1(raw_u8, out_features=out_features, in_features=in_features)

    torch.manual_seed(int(seed))
    device = torch.device("cuda")

    tile_k = TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1
    block_bytes = Q4K_BLOCK_BYTES
    k_tiles = in_features // tile_k

    raw_blocks_u8 = torch.from_numpy(np.array(raw_u8.reshape((out_features * k_tiles, block_bytes)), copy=True)).to(
        device=device, dtype=torch.uint8
    )
    w_blocks = dequantize_blocks_Q4_K(raw_blocks_u8, tile_k, block_bytes, dtype=torch.float16)
    w = w_blocks.reshape((out_features, k_tiles, tile_k)).reshape((out_features, in_features))

    x = torch.randn((int(batch), int(in_features)), device=device, dtype=torch.float16)
    bias_t = torch.randn((int(out_features),), device=device, dtype=torch.float16) if bias else None

    y_ref = F.linear(x, w, bias_t)
    packed_t = torch.from_numpy(np.array(packed_i8, copy=True)).to(device=device)
    y = torch.ops.codexpack.q4k_tilepack_linear(x, packed_t, int(out_features), int(in_features), bias_t)
    torch.cuda.synchronize()

    torch.testing.assert_close(y, y_ref, rtol=float(rtol), atol=float(atol))
    return float((y - y_ref).abs().max().detach().cpu().item())


def _build_cli(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate CodexPack q4k_tilepack_linear correctness on CUDA (SM86+).")
    p.add_argument("--out-features", type=int, default=256)
    p.add_argument("--in-features", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bias", dest="bias", action="store_false")
    p.set_defaults(bias=True)
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--atol", type=float, default=2e-2)
    return p.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    ns = _build_cli(argv)
    max_abs = validate_q4k_tilepack_linear(
        out_features=ns.out_features,
        in_features=ns.in_features,
        batch=ns.batch,
        seed=ns.seed,
        bias=bool(ns.bias),
        rtol=ns.rtol,
        atol=ns.atol,
    )
    write_stdout(f"OK: max_abs_err={max_abs:.6g}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
