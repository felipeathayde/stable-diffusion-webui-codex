"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z-Image transformer SafeTensors → GGUF conversion helpers.
Converts official Diffusers-style Z-Image transformer weights (separate Q/K/V projection tensors and `all_*` prefixes) into the Codex/Comfy
runtime layout:
- attention `to_{q,k,v}` → concatenated `qkv`
- attention `to_out.0` → `out`
- attention `norm_{q,k}` → `q_norm` / `k_norm`
- `all_x_embedder.*` → `x_embedder.*`
- `all_final_layer.*` → `final_layer.*`

This module is used by the generic GGUF converter when the provided `config.json` identifies a `ZImageTransformer2DModel`.

Symbols (top-level; keep in sync; no ghosts):
- `is_zimage_transformer_config` (function): Returns True when a loaded config.json represents the Z-Image transformer.
- `convert_zimage_transformer_to_gguf` (function): Convert a Z-Image transformer SafeTensors source into a GGUF file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from apps.backend.quantization.api import quantize_numpy
from apps.backend.quantization.gguf import GGMLQuantizationType, GGUFWriter
from apps.backend.quantization.gguf.quant_shapes import quant_shape_to_byte_shape
from apps.backend.runtime.tools import gguf_converter_metadata as _metadata
from apps.backend.runtime.tools import gguf_converter_quantization as _quantization
from apps.backend.runtime.tools import gguf_converter_safetensors_source as _safetensors_source
from apps.backend.runtime.tools import gguf_converter_verify as _verify
from apps.backend.runtime.tools.gguf_converter_tensor_planner import TensorPlan
from apps.backend.runtime.tools.gguf_converter_types import ConversionConfig, ConversionProgress, QuantizationType


def is_zimage_transformer_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("_class_name") or "") == "ZImageTransformer2DModel"


@dataclass(frozen=True, slots=True)
class _Op:
    kind: str  # "copy" | "concat_qkv"
    src_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ZPlan:
    gguf_name: str
    raw_shape: tuple[int, ...]
    ggml_type: GGMLQuantizationType
    stored_shape: tuple[int, ...]
    stored_dtype: np.dtype
    stored_nbytes: int
    op: _Op

    def to_verify_plan(self) -> TensorPlan:
        # Verification expects a single source key; for synthetic tensors use the first
        # input tensor as a representative for spot-check mapping.
        src = self.op.src_names[0] if self.op.src_names else self.gguf_name
        return TensorPlan(
            src_name=src,
            gguf_name=self.gguf_name,
            raw_shape=self.raw_shape,
            ggml_type=self.ggml_type,
            stored_shape=self.stored_shape,
            stored_dtype=self.stored_dtype,
            stored_nbytes=self.stored_nbytes,
        )


_RX_ALL_X_EMBEDDER = re.compile(r"^all_x_embedder\.[^.]+\.(?P<suffix>weight|bias)$")
_RX_ALL_FINAL_LAYER = re.compile(r"^all_final_layer\.[^.]+\.(?P<rest>.+)$")

_RX_ATTN_QKV = re.compile(r"^(?P<prefix>.+\.attention)\.to_(?P<which>[qkv])\.(?P<param>weight|bias)$")
_RX_ATTN_OUT = re.compile(r"^(?P<prefix>.+\.attention)\.to_out\.0\.(?P<param>weight|bias)$")
_RX_ATTN_NORM = re.compile(r"^(?P<prefix>.+\.attention)\.norm_(?P<which>[qk])\.weight$")

_PAD_TOKENS = {"x_pad_token", "cap_pad_token"}


def _normalize_zimage_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Z-Image transformer config keys into the metadata helper's expected fields."""

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    # `all_patch_size` is a list in HF configs; extract scalar for context_length-ish guesses.
    patch_size = config.get("all_patch_size")
    if isinstance(patch_size, list) and patch_size:
        patch_size = patch_size[0]
    patch = _as_int(patch_size, 2)

    dim = _as_int(config.get("dim"), 3840)
    n_layers = _as_int(config.get("n_layers"), 30)
    n_heads = _as_int(config.get("n_heads"), 30)
    n_kv = _as_int(config.get("n_kv_heads"), n_heads)

    axes_lens = config.get("axes_lens")
    if isinstance(axes_lens, list) and len(axes_lens) == 3:
        max_pos = max(_as_int(v, 0) for v in axes_lens)
    else:
        max_pos = 4096

    return {
        "model_type": "zimage",
        "num_hidden_layers": n_layers,
        "hidden_size": dim,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "max_position_embeddings": max_pos * max(1, patch),
        "rope_theta": _as_float(config.get("rope_theta"), 256.0),
        "rms_norm_eps": _as_float(config.get("norm_eps"), 1e-5),
        "_name_or_path": str(config.get("_name_or_path") or config.get("name") or "Alibaba-TongYi/Z-Image-Turbo"),
    }


def _select_ggml_type(
    *,
    raw_shape: tuple[int, ...],
    gguf_name: str,
    src_names: Sequence[str],
    requested: GGMLQuantizationType,
    overrides: list[tuple[re.Pattern[str], GGMLQuantizationType]],
) -> GGMLQuantizationType:
    desired = requested
    for rx, qtype in overrides:
        if rx.search(gguf_name):
            desired = qtype
            continue
        for src in src_names:
            if rx.search(src):
                desired = qtype
                break

    ggml_type = _quantization.select_tensor_ggml_type(raw_shape, desired)

    # Z-Image pad tokens are plain `nn.Parameter` tensors; quantizing them produces GGUF
    # packed-byte shapes that can't be loaded via `nn.Module.load_state_dict`.
    if gguf_name in _PAD_TOKENS and ggml_type not in {GGMLQuantizationType.F16, GGMLQuantizationType.F32}:
        return GGMLQuantizationType.F16

    return ggml_type


def _build_plan(
    *,
    gguf_name: str,
    raw_shape: tuple[int, ...],
    op: _Op,
    requested: GGMLQuantizationType,
    overrides: list[tuple[re.Pattern[str], GGMLQuantizationType]],
) -> _ZPlan:
    ggml_type = _select_ggml_type(
        raw_shape=raw_shape,
        gguf_name=gguf_name,
        src_names=op.src_names,
        requested=requested,
        overrides=overrides,
    )

    if ggml_type == GGMLQuantizationType.F16:
        stored_dtype = np.dtype(np.float16)
        stored_shape = raw_shape
        stored_nbytes = int(np.prod(raw_shape, dtype=np.int64) * 2)
    elif ggml_type == GGMLQuantizationType.F32:
        stored_dtype = np.dtype(np.float32)
        stored_shape = raw_shape
        stored_nbytes = int(np.prod(raw_shape, dtype=np.int64) * 4)
    else:
        stored_dtype = np.dtype(np.uint8)
        stored_shape = quant_shape_to_byte_shape(raw_shape, ggml_type)
        stored_nbytes = int(np.prod(stored_shape, dtype=np.int64))

    return _ZPlan(
        gguf_name=gguf_name,
        raw_shape=raw_shape,
        ggml_type=ggml_type,
        stored_shape=stored_shape,
        stored_dtype=stored_dtype,
        stored_nbytes=stored_nbytes,
        op=op,
    )


def _shape_of(source: Any, cache: dict[str, tuple[int, ...]], name: str) -> tuple[int, ...]:
    cached = cache.get(name)
    if cached is not None:
        return cached
    sl = source.get_slice(name)
    shape = tuple(int(x) for x in sl.get_shape())
    cache[name] = shape
    return shape


def convert_zimage_transformer_to_gguf(
    config: ConversionConfig,
    *,
    model_config: Mapping[str, Any],
    config_path: Path,
    progress: ConversionProgress,
    update_progress: Callable[[], None],
) -> str:
    requested_type = _quantization.requested_ggml_type(config.quantization)
    overrides = _quantization.compile_tensor_overrides(config.quantization, config.tensor_type_overrides)

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    progress.status = "loading_weights"
    update_progress()

    z_meta_cfg = _normalize_zimage_metadata_config(model_config)
    arch = str(z_meta_cfg.get("model_type") or "zimage")

    with _safetensors_source.open_safetensors_source(config.safetensors_path) as source:
        tensor_names = list(source.keys())

        shapes: dict[str, tuple[int, ...]] = {}
        out_specs: dict[str, _Op] = {}
        qkv_groups: dict[str, dict[str, dict[str, str]]] = {}

        # 1) First pass: build direct key remaps and collect Q/K/V groups.
        for src_name in tensor_names:
            m = _RX_ALL_X_EMBEDDER.match(src_name)
            if m:
                suffix = m.group("suffix")
                out_specs[f"x_embedder.{suffix}"] = _Op(kind="copy", src_names=(src_name,))
                continue

            m = _RX_ALL_FINAL_LAYER.match(src_name)
            if m:
                rest = m.group("rest")
                out_specs[f"final_layer.{rest}"] = _Op(kind="copy", src_names=(src_name,))
                continue

            m = _RX_ATTN_OUT.match(src_name)
            if m:
                prefix = m.group("prefix")
                param = m.group("param")
                out_specs[f"{prefix}.out.{param}"] = _Op(kind="copy", src_names=(src_name,))
                continue

            m = _RX_ATTN_NORM.match(src_name)
            if m:
                prefix = m.group("prefix")
                which = m.group("which")
                out_specs[f"{prefix}.{'q_norm' if which == 'q' else 'k_norm'}.weight"] = _Op(
                    kind="copy",
                    src_names=(src_name,),
                )
                continue

            m = _RX_ATTN_QKV.match(src_name)
            if m:
                prefix = m.group("prefix")
                which = m.group("which")
                param = m.group("param")
                entry = qkv_groups.setdefault(prefix, {}).setdefault(param, {})
                entry[which] = src_name
                continue

            # Default: keep key as-is.
            out_specs[src_name] = _Op(kind="copy", src_names=(src_name,))

        # 2) Synthesize concatenated QKV tensors and validate inputs.
        for prefix, per_param in sorted(qkv_groups.items()):
            weights = per_param.get("weight") or {}
            missing = [k for k in ("q", "k", "v") if k not in weights]
            if missing:
                raise RuntimeError(
                    f"ZImage converter: missing attention.to_* weights for {prefix}: {missing} "
                    f"(present={sorted(weights)})"
                )

            q_shape = _shape_of(source, shapes, weights["q"])
            k_shape = _shape_of(source, shapes, weights["k"])
            v_shape = _shape_of(source, shapes, weights["v"])
            if q_shape != k_shape or q_shape != v_shape or len(q_shape) != 2:
                raise RuntimeError(
                    "ZImage converter: Q/K/V weight shapes must match and be 2D for "
                    f"{prefix}: q={q_shape} k={k_shape} v={v_shape}"
                )

            qkv_shape = (int(q_shape[0]) * 3, int(q_shape[1]))
            out_specs[f"{prefix}.qkv.weight"] = _Op(
                kind="concat_qkv",
                src_names=(weights["q"], weights["k"], weights["v"]),
            )

            # Bias is optional, but if present it must be complete.
            biases = per_param.get("bias") or {}
            if biases:
                missing_b = [k for k in ("q", "k", "v") if k not in biases]
                if missing_b:
                    raise RuntimeError(
                        f"ZImage converter: missing attention.to_* biases for {prefix}: {missing_b} "
                        f"(present={sorted(biases)})"
                    )
                qb = _shape_of(source, shapes, biases["q"])
                kb = _shape_of(source, shapes, biases["k"])
                vb = _shape_of(source, shapes, biases["v"])
                if qb != kb or qb != vb or len(qb) != 1:
                    raise RuntimeError(
                        "ZImage converter: Q/K/V bias shapes must match and be 1D for "
                        f"{prefix}: q={qb} k={kb} v={vb}"
                    )
                out_specs[f"{prefix}.qkv.bias"] = _Op(
                    kind="concat_qkv",
                    src_names=(biases["q"], biases["k"], biases["v"]),
                )

        # 3) Turn output specs into deterministic plans.
        plans: list[_ZPlan] = []
        for gguf_name in sorted(out_specs):
            op = out_specs[gguf_name]
            if op.kind == "copy":
                raw_shape = _shape_of(source, shapes, op.src_names[0])
            elif op.kind == "concat_qkv":
                # Weight concat uses 2D tensors; bias concat uses 1D. Use the first input
                # tensor to infer base shape and expand.
                base_shape = _shape_of(source, shapes, op.src_names[0])
                if len(base_shape) == 2:
                    raw_shape = (int(base_shape[0]) * 3, int(base_shape[1]))
                elif len(base_shape) == 1:
                    raw_shape = (int(base_shape[0]) * 3,)
                else:
                    raise RuntimeError(f"ZImage converter: unexpected concat_qkv tensor shape for {gguf_name}: {base_shape}")
            else:  # pragma: no cover - defensive
                raise RuntimeError(f"ZImage converter: unknown op {op.kind!r} for {gguf_name}")

            plans.append(
                _build_plan(
                    gguf_name=gguf_name,
                    raw_shape=raw_shape,
                    op=op,
                    requested=requested_type,
                    overrides=overrides,
                )
            )

        progress.total_steps = len(plans)
        progress.status = "converting"
        update_progress()

        # Map a subset of source keys to output keys for verification spot-checks.
        key_mapping: dict[str, str] = {}
        for plan in plans:
            if plan.op.kind == "copy":
                key_mapping[plan.op.src_names[0]] = plan.gguf_name
            elif plan.op.kind == "concat_qkv":
                # Only map the Q tensor so spot-check verifies the first rows.
                key_mapping[plan.op.src_names[0]] = plan.gguf_name

        writer = GGUFWriter(path=str(output_path), arch=arch)
        _metadata.add_basic_metadata(
            writer,
            arch,
            z_meta_cfg,
            config.quantization,
            config_path=config_path,
            safetensors_path=config.safetensors_path,
        )

        for plan in plans:
            raw_dtype = None if plan.ggml_type in {GGMLQuantizationType.F16, GGMLQuantizationType.F32} else plan.ggml_type
            writer.add_tensor_info(
                plan.gguf_name,
                tensor_shape=plan.stored_shape,
                tensor_dtype=plan.stored_dtype,
                tensor_nbytes=plan.stored_nbytes,
                raw_dtype=raw_dtype,
            )

        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()

            assert writer.fout is not None
            out = writer.fout[0]
            writer.write_padding(out, out.tell())

            chunk_rows = 1024
            for idx, plan in enumerate(plans):
                progress.current_step = idx + 1
                progress.current_tensor = plan.gguf_name
                update_progress()

                bytes_written = 0

                if plan.op.kind == "copy":
                    src_name = plan.op.src_names[0]
                    sl = source.get_slice(src_name)
                    shape = tuple(int(x) for x in sl.get_shape())
                    if shape != plan.raw_shape:
                        raise RuntimeError(
                            f"Tensor shape changed during conversion for {src_name}: {shape} vs {plan.raw_shape}"
                        )

                    if plan.ggml_type == GGMLQuantizationType.F16:
                        target_dtype = torch.float16
                        if len(shape) == 1:
                            t = sl[:].to(target_dtype).contiguous()
                            out.write(t.numpy().tobytes(order="C"))
                            bytes_written += t.numel() * 2
                        elif len(shape) == 2:
                            rows = shape[0]
                            for start in range(0, rows, chunk_rows):
                                chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                                out.write(chunk.numpy().tobytes(order="C"))
                                bytes_written += chunk.numel() * 2
                        else:
                            t = source.get_tensor(src_name).to(target_dtype).contiguous()
                            out.write(t.numpy().tobytes(order="C"))
                            bytes_written += t.numel() * 2

                    elif plan.ggml_type == GGMLQuantizationType.F32:
                        target_dtype = torch.float32
                        if len(shape) == 1:
                            t = sl[:].to(target_dtype).contiguous()
                            out.write(t.numpy().tobytes(order="C"))
                            bytes_written += t.numel() * 4
                        elif len(shape) == 2:
                            rows = shape[0]
                            for start in range(0, rows, chunk_rows):
                                chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                                out.write(chunk.numpy().tobytes(order="C"))
                                bytes_written += chunk.numel() * 4
                        else:
                            t = source.get_tensor(src_name).to(target_dtype).contiguous()
                            out.write(t.numpy().tobytes(order="C"))
                            bytes_written += t.numel() * 4

                    else:
                        if len(shape) == 1:
                            raise RuntimeError(f"Unexpected quantized 1D tensor plan for {src_name}: {shape}")

                        if len(shape) == 2:
                            rows = shape[0]
                            for start in range(0, rows, chunk_rows):
                                chunk = sl[start : min(rows, start + chunk_rows)].to(torch.float32).contiguous()
                                arr = chunk.numpy()
                                q = quantize_numpy(arr, plan.ggml_type)
                                out.write(q.tobytes(order="C"))
                                bytes_written += q.nbytes
                        else:
                            t = source.get_tensor(src_name).to(torch.float32).contiguous()
                            q = quantize_numpy(t.numpy(), plan.ggml_type)
                            out.write(q.tobytes(order="C"))
                            bytes_written += q.nbytes

                elif plan.op.kind == "concat_qkv":
                    src_q, src_k, src_v = plan.op.src_names
                    q_sl = source.get_slice(src_q)
                    k_sl = source.get_slice(src_k)
                    v_sl = source.get_slice(src_v)
                    q_shape = tuple(int(x) for x in q_sl.get_shape())
                    k_shape = tuple(int(x) for x in k_sl.get_shape())
                    v_shape = tuple(int(x) for x in v_sl.get_shape())
                    if q_shape != k_shape or q_shape != v_shape:
                        raise RuntimeError(
                            f"ZImage converter: concat_qkv shape mismatch for {plan.gguf_name}: "
                            f"q={q_shape} k={k_shape} v={v_shape}"
                        )

                    if len(q_shape) == 1:
                        # Bias concat: always store as float (1D tensors are F16/F32).
                        if plan.ggml_type not in {GGMLQuantizationType.F16, GGMLQuantizationType.F32}:
                            raise RuntimeError(f"Unexpected quantized bias plan for {plan.gguf_name}: {plan.ggml_type.name}")
                        target_dtype = torch.float16 if plan.ggml_type == GGMLQuantizationType.F16 else torch.float32
                        t = torch.cat(
                            [q_sl[:].to(target_dtype), k_sl[:].to(target_dtype), v_sl[:].to(target_dtype)],
                            dim=0,
                        ).contiguous()
                        out.write(t.numpy().tobytes(order="C"))
                        bytes_written += t.numel() * (2 if target_dtype == torch.float16 else 4)
                    else:
                        rows = q_shape[0]
                        cols = q_shape[1]
                        if plan.raw_shape != (rows * 3, cols):
                            raise RuntimeError(
                                f"ZImage converter: concat_qkv raw_shape mismatch for {plan.gguf_name}: "
                                f"expected={(rows * 3, cols)} planned={plan.raw_shape}"
                            )

                        if plan.ggml_type == GGMLQuantizationType.F16:
                            target_dtype = torch.float16
                            for sl in (q_sl, k_sl, v_sl):
                                for start in range(0, rows, chunk_rows):
                                    chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                                    out.write(chunk.numpy().tobytes(order="C"))
                                    bytes_written += chunk.numel() * 2
                        elif plan.ggml_type == GGMLQuantizationType.F32:
                            target_dtype = torch.float32
                            for sl in (q_sl, k_sl, v_sl):
                                for start in range(0, rows, chunk_rows):
                                    chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                                    out.write(chunk.numpy().tobytes(order="C"))
                                    bytes_written += chunk.numel() * 4
                        else:
                            for sl in (q_sl, k_sl, v_sl):
                                for start in range(0, rows, chunk_rows):
                                    chunk = sl[start : min(rows, start + chunk_rows)].to(torch.float32).contiguous()
                                    q = quantize_numpy(chunk.numpy(), plan.ggml_type)
                                    out.write(q.tobytes(order="C"))
                                    bytes_written += q.nbytes

                else:  # pragma: no cover - defensive
                    raise RuntimeError(f"ZImage converter: unknown op {plan.op.kind!r}")

                if bytes_written != plan.stored_nbytes:
                    raise RuntimeError(
                        f"Byte count mismatch for {plan.gguf_name}: wrote {bytes_written}, expected {plan.stored_nbytes}"
                    )
                writer.write_padding(out, plan.stored_nbytes)
        finally:
            writer.close()

    progress.status = "verifying"
    update_progress()

    verify_plans = [p.to_verify_plan() for p in plans]
    _verify.verify_gguf_file(
        gguf_path=str(output_path),
        source_safetensors=config.safetensors_path,
        tensor_plans=verify_plans,
        key_mapping=key_mapping,
    )

    progress.status = "complete"
    progress.current_step = progress.total_steps
    update_progress()
    return str(output_path)


__all__ = [
    "convert_zimage_transformer_to_gguf",
    "is_zimage_transformer_config",
]

