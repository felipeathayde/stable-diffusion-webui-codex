"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tensor planning helpers for the GGUF converter.
Plans tensor name remaps, quantization types, and storage byte shapes without loading full tensors.

Symbols (top-level; keep in sync; no ghosts):
- `TensorPlan` (dataclass): Planned tensor conversion entry (name/shape/type + storage strategy).
- `plan_tensors` (function): Plan per-tensor conversion settings for a safetensors source.
- `is_zimage_transformer_config` (function): Returns True when a config.json represents a Z-Image transformer export.
- `normalize_zimage_transformer_metadata_config` (function): Adapts Z-Image transformer config fields to metadata helper inputs.
- `plan_zimage_transformer_tensors` (function): Plan tensor conversion for Diffusers-style Z-Image transformer weights (includes QKV packing).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from apps.backend.quantization.gguf import GGMLQuantizationType
from apps.backend.quantization.gguf.quant_shapes import quant_shape_to_byte_shape
from apps.backend.runtime.tools.gguf_converter_quantization import select_tensor_ggml_type


@dataclass(frozen=True, slots=True)
class TensorPlan:
    src_name: str
    gguf_name: str
    raw_shape: tuple[int, ...]
    ggml_type: GGMLQuantizationType
    stored_shape: tuple[int, ...]
    stored_dtype: np.dtype
    stored_nbytes: int
    op: str
    src_names: tuple[str, ...]


def plan_tensors(
    tensor_names: list[str],
    safetensors_handle: Any,
    key_mapping: dict[str, str],
    requested: GGMLQuantizationType,
    overrides: list[tuple[re.Pattern[str], GGMLQuantizationType]],
) -> list[TensorPlan]:
    plans: list[TensorPlan] = []

    for src_name in tensor_names:
        sl = safetensors_handle.get_slice(src_name)
        raw_shape = tuple(int(x) for x in sl.get_shape())
        gguf_name = key_mapping.get(src_name, src_name)

        desired = requested
        for rx, qtype in overrides:
            if rx.search(src_name) or rx.search(gguf_name):
                desired = qtype
        ggml_type = select_tensor_ggml_type(raw_shape, desired)

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

        plans.append(
            TensorPlan(
                src_name=src_name,
                gguf_name=gguf_name,
                raw_shape=raw_shape,
                ggml_type=ggml_type,
                stored_shape=stored_shape,
                stored_dtype=stored_dtype,
                stored_nbytes=stored_nbytes,
                op="copy",
                src_names=(src_name,),
            )
        )

    return plans


def is_zimage_transformer_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("_class_name") or "") == "ZImageTransformer2DModel"


def normalize_zimage_transformer_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Z-Image transformer config keys into the metadata helper's expected fields.

    The GGUF metadata helper is LLM-shaped (hidden_size, num_hidden_layers, etc.).
    Z-Image transformer configs use Diffusers keys (`dim`, `n_layers`, ...).
    """

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


_RX_ALL_X_EMBEDDER = re.compile(r"^all_x_embedder\.[^.]+\.(?P<suffix>weight|bias)$")
_RX_ALL_FINAL_LAYER = re.compile(r"^all_final_layer\.[^.]+\.(?P<rest>.+)$")

_RX_ATTN_QKV = re.compile(r"^(?P<prefix>.+\.attention)\.to_(?P<which>[qkv])\.(?P<param>weight|bias)$")
_RX_ATTN_OUT = re.compile(r"^(?P<prefix>.+\.attention)\.to_out\.0\.(?P<param>weight|bias)$")
_RX_ATTN_NORM = re.compile(r"^(?P<prefix>.+\.attention)\.norm_(?P<which>[qk])\.weight$")

_ZIMAGE_PAD_TOKENS = {"x_pad_token", "cap_pad_token"}


def _shape_of(safetensors_handle: Any, cache: dict[str, tuple[int, ...]], name: str) -> tuple[int, ...]:
    cached = cache.get(name)
    if cached is not None:
        return cached
    sl = safetensors_handle.get_slice(name)
    shape = tuple(int(x) for x in sl.get_shape())
    cache[name] = shape
    return shape


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

    ggml_type = select_tensor_ggml_type(raw_shape, desired)

    # Z-Image pad tokens are plain nn.Parameters; quantizing them produces packed-byte
    # shapes that can't be loaded via nn.Module.load_state_dict.
    if gguf_name in _ZIMAGE_PAD_TOKENS and ggml_type not in {GGMLQuantizationType.F16, GGMLQuantizationType.F32}:
        return GGMLQuantizationType.F16

    return ggml_type


def _build_plan(
    *,
    gguf_name: str,
    raw_shape: tuple[int, ...],
    op: str,
    src_name: str,
    src_names: tuple[str, ...],
    requested: GGMLQuantizationType,
    overrides: list[tuple[re.Pattern[str], GGMLQuantizationType]],
) -> TensorPlan:
    ggml_type = _select_ggml_type(
        raw_shape=raw_shape,
        gguf_name=gguf_name,
        src_names=src_names,
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

    return TensorPlan(
        src_name=src_name,
        gguf_name=gguf_name,
        raw_shape=raw_shape,
        ggml_type=ggml_type,
        stored_shape=stored_shape,
        stored_dtype=stored_dtype,
        stored_nbytes=stored_nbytes,
        op=op,
        src_names=src_names,
    )


def plan_zimage_transformer_tensors(
    tensor_names: list[str],
    safetensors_handle: Any,
    requested: GGMLQuantizationType,
    overrides: list[tuple[re.Pattern[str], GGMLQuantizationType]],
) -> tuple[list[TensorPlan], dict[str, str]]:
    """Plan Z-Image transformer tensors from Diffusers key layout into Codex runtime layout.

    Includes attention QKV packing (to_q/to_k/to_v → qkv) and basic key renames for
    the official sharded Diffusers exports.
    """

    shapes: dict[str, tuple[int, ...]] = {}
    consumed: set[str] = set()

    out_ops: dict[str, tuple[str, tuple[str, ...]]] = {}

    # 1) One-to-one renames.
    for src_name in tensor_names:
        m = _RX_ALL_X_EMBEDDER.match(src_name)
        if m:
            suffix = m.group("suffix")
            out_key = f"x_embedder.{suffix}"
            if out_key in out_ops:
                raise RuntimeError(f"ZImage planner: duplicate mapping for {out_key} (src={src_name})")
            out_ops[out_key] = ("copy", (src_name,))
            consumed.add(src_name)
            continue

        m = _RX_ALL_FINAL_LAYER.match(src_name)
        if m:
            rest = m.group("rest")
            out_key = f"final_layer.{rest}"
            if out_key in out_ops:
                raise RuntimeError(f"ZImage planner: duplicate mapping for {out_key} (src={src_name})")
            out_ops[out_key] = ("copy", (src_name,))
            consumed.add(src_name)
            continue

        m = _RX_ATTN_OUT.match(src_name)
        if m:
            prefix = m.group("prefix")
            param = m.group("param")
            out_key = f"{prefix}.out.{param}"
            if out_key in out_ops:
                raise RuntimeError(f"ZImage planner: duplicate mapping for {out_key} (src={src_name})")
            out_ops[out_key] = ("copy", (src_name,))
            consumed.add(src_name)
            continue

        m = _RX_ATTN_NORM.match(src_name)
        if m:
            prefix = m.group("prefix")
            which = m.group("which")
            out_key = f"{prefix}.{'q_norm' if which == 'q' else 'k_norm'}.weight"
            if out_key in out_ops:
                raise RuntimeError(f"ZImage planner: duplicate mapping for {out_key} (src={src_name})")
            out_ops[out_key] = ("copy", (src_name,))
            consumed.add(src_name)
            continue

    # 2) Collect split Q/K/V projections for fusion.
    qkv_groups: dict[str, dict[str, dict[str, str]]] = {}
    for src_name in tensor_names:
        m = _RX_ATTN_QKV.match(src_name)
        if not m:
            continue
        prefix = m.group("prefix")
        which = m.group("which")
        param = m.group("param")
        qkv_groups.setdefault(prefix, {}).setdefault(param, {})[which] = src_name

    for prefix, per_param in sorted(qkv_groups.items()):
        weights = per_param.get("weight") or {}
        missing = [k for k in ("q", "k", "v") if k not in weights]
        if missing:
            raise RuntimeError(
                f"ZImage planner: missing attention.to_* weights for {prefix}: {missing} (present={sorted(weights)})"
            )

        q_shape = _shape_of(safetensors_handle, shapes, weights["q"])
        k_shape = _shape_of(safetensors_handle, shapes, weights["k"])
        v_shape = _shape_of(safetensors_handle, shapes, weights["v"])
        if q_shape != k_shape or q_shape != v_shape or len(q_shape) != 2:
            raise RuntimeError(
                "ZImage planner: Q/K/V weight shapes must match and be 2D for "
                f"{prefix}: q={q_shape} k={k_shape} v={v_shape}"
            )

        out_ops[f"{prefix}.qkv.weight"] = ("concat_dim0", (weights["q"], weights["k"], weights["v"]))
        consumed.update((weights["q"], weights["k"], weights["v"]))

        biases = per_param.get("bias") or {}
        if biases:
            missing_b = [k for k in ("q", "k", "v") if k not in biases]
            if missing_b:
                raise RuntimeError(
                    f"ZImage planner: missing attention.to_* biases for {prefix}: {missing_b} (present={sorted(biases)})"
                )
            qb = _shape_of(safetensors_handle, shapes, biases["q"])
            kb = _shape_of(safetensors_handle, shapes, biases["k"])
            vb = _shape_of(safetensors_handle, shapes, biases["v"])
            if qb != kb or qb != vb or len(qb) != 1:
                raise RuntimeError(
                    "ZImage planner: Q/K/V bias shapes must match and be 1D for "
                    f"{prefix}: q={qb} k={kb} v={vb}"
                )
            out_ops[f"{prefix}.qkv.bias"] = ("concat_dim0", (biases["q"], biases["k"], biases["v"]))
            consumed.update((biases["q"], biases["k"], biases["v"]))

    # 3) Pass-through any remaining tensors.
    for src_name in tensor_names:
        if src_name in consumed:
            continue
        if src_name in out_ops:
            continue
        out_ops[src_name] = ("copy", (src_name,))

    # 4) Materialize output plans + verification mapping.
    plans: list[TensorPlan] = []
    key_mapping: dict[str, str] = {}

    for gguf_name in sorted(out_ops):
        op, srcs = out_ops[gguf_name]
        if not srcs:
            raise RuntimeError(f"ZImage planner: empty source list for {gguf_name}")

        if op == "copy":
            raw_shape = _shape_of(safetensors_handle, shapes, srcs[0])
            src_name = srcs[0]
        elif op == "concat_dim0":
            base_shape = _shape_of(safetensors_handle, shapes, srcs[0])
            if len(base_shape) == 2:
                raw_shape = (int(base_shape[0]) * len(srcs), int(base_shape[1]))
            elif len(base_shape) == 1:
                raw_shape = (int(base_shape[0]) * len(srcs),)
            else:
                raise RuntimeError(f"ZImage planner: unexpected concat_dim0 source shape for {gguf_name}: {base_shape}")
            src_name = srcs[0]
        else:
            raise RuntimeError(f"ZImage planner: unknown op={op!r} for {gguf_name}")

        plans.append(
            _build_plan(
                gguf_name=gguf_name,
                raw_shape=raw_shape,
                op=op,
                src_name=src_name,
                src_names=tuple(srcs),
                requested=requested,
                overrides=overrides,
            )
        )

        # Provide a stable mapping for verification spot-checks.
        if src_name not in key_mapping:
            key_mapping[src_name] = gguf_name

    return plans, key_mapping


__all__ = [
    "TensorPlan",
    "is_zimage_transformer_config",
    "normalize_zimage_transformer_metadata_config",
    "plan_tensors",
    "plan_zimage_transformer_tensors",
]
