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
- `is_flux_transformer_config` (function): Returns True when a config.json represents a Flux transformer export.
- `normalize_flux_transformer_metadata_config` (function): Adapts Flux transformer config fields to metadata helper inputs.
- `plan_flux_transformer_tensors` (function): Plan tensor conversion for Diffusers-style Flux transformer weights (maps to Comfy-style keys).
- `is_wan22_transformer_config` (function): Returns True when a config.json represents a WAN22 transformer export.
- `normalize_wan22_transformer_metadata_config` (function): Adapts WAN22 transformer config fields to metadata helper inputs.
- `plan_wan22_transformer_tensors` (function): Plan tensor conversion for WAN22 transformer weights (Diffusers → Comfy key mapping).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from apps.backend.quantization.gguf import GGMLQuantizationType
from apps.backend.quantization.gguf.quant_shapes import quant_shape_to_byte_shape
from apps.backend.runtime.tools.gguf_converter_quantization import select_tensor_ggml_type
from apps.backend.runtime.tools.gguf_converter_specs import CompiledTensorTypeRule


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
    overrides: list[CompiledTensorTypeRule],
) -> list[TensorPlan]:
    plans: list[TensorPlan] = []

    for src_name in tensor_names:
        sl = safetensors_handle.get_slice(src_name)
        raw_shape = tuple(int(x) for x in sl.get_shape())
        gguf_name = key_mapping.get(src_name, src_name)

        desired = requested
        for rule in overrides:
            if rule.apply_to.matches_src() and rule.pattern.search(src_name):
                desired = rule.ggml_type
            if rule.apply_to.matches_dst() and rule.pattern.search(gguf_name):
                desired = rule.ggml_type
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


def is_flux_transformer_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("_class_name") or "") == "FluxTransformer2DModel"


def is_wan22_transformer_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("_class_name") or "") in {"WanTransformer3DModel", "WanModel"}


def normalize_flux_transformer_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Flux transformer config keys into the metadata helper's expected fields."""

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    num_heads = _as_int(config.get("num_attention_heads"), 24)
    head_dim = _as_int(config.get("attention_head_dim"), 128)
    hidden = max(1, num_heads) * max(1, head_dim)

    double_layers = _as_int(config.get("num_layers"), 19)
    single_layers = _as_int(config.get("num_single_layers"), 38)

    return {
        "model_type": "flux",
        "num_hidden_layers": max(1, double_layers + single_layers),
        "hidden_size": hidden,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "_name_or_path": str(
            config.get("_name_or_path") or config.get("name") or "black-forest-labs/FLUX.1-Kontext-dev"
        ),
    }


def normalize_wan22_transformer_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt WAN22 transformer config keys into the metadata helper's expected fields."""

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

    class_name = str(config.get("_class_name") or "")
    if class_name == "WanModel":
        hidden = _as_int(config.get("dim"), 4096)
        num_layers = _as_int(config.get("num_layers"), 32)
        num_heads = _as_int(config.get("num_heads"), 32)
        max_pos = _as_int(config.get("text_len"), 512)
        eps = _as_float(config.get("eps"), 1e-6)
        model_type = str(config.get("model_type") or "").strip().lower()
        name = "Wan-AI/Wan2.2"
        if model_type == "i2v":
            name = "Wan-AI/Wan2.2-I2V-A14B"
        elif model_type == "t2v":
            name = "Wan-AI/Wan2.2-T2V-A14B"
        return {
            "model_type": "wan22",
            "num_hidden_layers": max(1, num_layers),
            "hidden_size": max(1, hidden),
            "num_attention_heads": max(1, num_heads),
            "num_key_value_heads": max(1, num_heads),
            "max_position_embeddings": max(1, max_pos),
            "rope_theta": 10000.0,
            "rms_norm_eps": eps,
            "_name_or_path": str(config.get("_name_or_path") or config.get("name") or name),
        }

    num_heads = _as_int(config.get("num_attention_heads"), 40)
    head_dim = _as_int(config.get("attention_head_dim"), 128)
    hidden = max(1, num_heads) * max(1, head_dim)

    num_layers = _as_int(config.get("num_layers"), 40)
    max_pos = _as_int(config.get("rope_max_seq_len"), 1024)
    eps = _as_float(config.get("eps"), 1e-6)

    name = "Wan-AI/Wan2.2"
    in_channels = _as_int(config.get("in_channels"), 16)
    if in_channels == 36:
        name = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    elif in_channels == 16:
        name = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    return {
        "model_type": "wan22",
        "num_hidden_layers": max(1, num_layers),
        "hidden_size": hidden,
        "num_attention_heads": max(1, num_heads),
        "num_key_value_heads": max(1, num_heads),
        "max_position_embeddings": max(1, max_pos),
        "rope_theta": 10000.0,
        "rms_norm_eps": eps,
        "_name_or_path": str(config.get("_name_or_path") or config.get("name") or name),
    }


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

_WAN22_TRANSFORMER_PREFIXES = (
    "model.diffusion_model.",
    "model.model.diffusion_model.",
    "model.",
)

_RX_WAN22_BLOCK_ATTN = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>attn1|attn2)\.to_(?P<proj>[qkv])\.(?P<param>weight|bias)$"
)
_RX_WAN22_BLOCK_ATTN_OUT = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>attn1|attn2)\.to_out\.0\.(?P<param>weight|bias)$"
)
_RX_WAN22_BLOCK_ATTN_NORM = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>attn1|attn2)\.norm_(?P<norm>[qk])\.weight$"
)
_RX_WAN22_BLOCK_FFN_PROJ = re.compile(
    r"^blocks\.(?P<idx>\d+)\.ffn\.net\.(?P<which>0\.proj|2)\.(?P<param>weight|bias)$"
)
_RX_WAN22_BLOCK_NORM2 = re.compile(r"^blocks\.(?P<idx>\d+)\.norm2\.(?P<param>weight|bias)$")
_RX_WAN22_BLOCK_SCALE_SHIFT = re.compile(r"^blocks\.(?P<idx>\d+)\.scale_shift_table$")


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
    overrides: list[CompiledTensorTypeRule],
) -> GGMLQuantizationType:
    desired = requested
    for rule in overrides:
        if rule.apply_to.matches_dst() and rule.pattern.search(gguf_name):
            desired = rule.ggml_type
            continue
        if rule.apply_to.matches_src():
            for src in src_names:
                if rule.pattern.search(src):
                    desired = rule.ggml_type
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
    overrides: list[CompiledTensorTypeRule],
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
    overrides: list[CompiledTensorTypeRule],
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


def _strip_prefixes(name: str, prefixes: tuple[str, ...]) -> str:
    out = name
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if out.startswith(prefix):
                out = out[len(prefix) :]
                changed = True
                break
    return out


def _map_wan22_key_to_comfy(src: str) -> str:
    key = _strip_prefixes(src, _WAN22_TRANSFORMER_PREFIXES)

    for before, after in (
        ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
        ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
        ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
        ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
        ("condition_embedder.time_proj.", "time_projection.1."),
        ("proj_out.", "head.head."),
    ):
        if key.startswith(before):
            return after + key[len(before) :]

    if key == "scale_shift_table":
        return "head.modulation"

    m = _RX_WAN22_BLOCK_ATTN.match(key)
    if m:
        idx = m.group("idx")
        which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
        proj = m.group("proj")
        param = m.group("param")
        return f"blocks.{idx}.{which}.{proj}.{param}"

    m = _RX_WAN22_BLOCK_ATTN_OUT.match(key)
    if m:
        idx = m.group("idx")
        which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
        param = m.group("param")
        return f"blocks.{idx}.{which}.o.{param}"

    m = _RX_WAN22_BLOCK_ATTN_NORM.match(key)
    if m:
        idx = m.group("idx")
        which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
        norm = m.group("norm")
        return f"blocks.{idx}.{which}.norm_{norm}.weight"

    m = _RX_WAN22_BLOCK_FFN_PROJ.match(key)
    if m:
        idx = m.group("idx")
        which = "0" if m.group("which") == "0.proj" else "2"
        param = m.group("param")
        return f"blocks.{idx}.ffn.{which}.{param}"

    m = _RX_WAN22_BLOCK_NORM2.match(key)
    if m:
        idx = m.group("idx")
        param = m.group("param")
        return f"blocks.{idx}.norm3.{param}"

    m = _RX_WAN22_BLOCK_SCALE_SHIFT.match(key)
    if m:
        idx = m.group("idx")
        return f"blocks.{idx}.modulation"

    return key


def plan_wan22_transformer_tensors(
    tensor_names: list[str],
    safetensors_handle: Any,
    requested: GGMLQuantizationType,
    overrides: list[CompiledTensorTypeRule],
) -> tuple[list[TensorPlan], dict[str, str]]:
    """Plan WAN22 transformer tensors.

    When inputs use Diffusers layout keys (`condition_embedder.*`, `attn1/attn2`, `ffn.net.*`, `scale_shift_table`),
    this normalizes them into the Comfy/WAN export layout keys used by Codex runtimes.
    """

    shapes: dict[str, tuple[int, ...]] = {}
    plans: list[TensorPlan] = []
    key_mapping: dict[str, str] = {}
    claimed: dict[str, str] = {}

    for src_name in tensor_names:
        gguf_name = _map_wan22_key_to_comfy(src_name)
        previous = claimed.get(gguf_name)
        if previous is not None and previous != src_name:
            raise RuntimeError(
                f"WAN22 planner: multiple source tensors map to the same output name: {gguf_name} ({previous}, {src_name})"
            )
        claimed[gguf_name] = src_name

        raw_shape = _shape_of(safetensors_handle, shapes, src_name)
        plans.append(
            _build_plan(
                gguf_name=gguf_name,
                raw_shape=raw_shape,
                op="copy",
                src_name=src_name,
                src_names=(src_name,),
                requested=requested,
                overrides=overrides,
            )
        )
        key_mapping[src_name] = gguf_name

    return plans, key_mapping


def _extract_block_indices(keys: Sequence[str], prefix: str) -> list[int]:
    indices: set[int] = set()
    for k in keys:
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix) :]
        part = rest.split(".", 1)[0]
        if part.isdigit():
            indices.add(int(part))
    return sorted(indices)


def plan_flux_transformer_tensors(
    tensor_names: list[str],
    safetensors_handle: Any,
    requested: GGMLQuantizationType,
    overrides: list[CompiledTensorTypeRule],
) -> tuple[list[TensorPlan], dict[str, str]]:
    """Plan Flux transformer tensors from Diffusers key layout into Comfy-style keys.

    This maps `transformer_blocks.*` (double blocks) and `single_transformer_blocks.*` (single blocks) into the
    key layout expected by `apps/backend/runtime/flux/model.py`:
    - `img_in`, `txt_in`, `time_in`, `vector_in`, `guidance_in`, `final_layer`
    - `double_blocks.*` (img/txt attn+mlp+modulation)
    - `single_blocks.*` (fused qkv+mlp in `linear1`, `linear2`, modulation)

    It also fuses QKV projections (and single-block `proj_mlp`) to match the fused Comfy layout.
    """

    shapes: dict[str, tuple[int, ...]] = {}
    keys_set = set(tensor_names)
    consumed: set[str] = set()
    out_ops: dict[str, tuple[str, tuple[str, ...]]] = {}

    def _require(name: str) -> str:
        if name not in keys_set:
            raise RuntimeError(f"Flux planner: missing required tensor: {name}")
        return name

    def _add_copy(dst: str, src: str) -> None:
        if dst in out_ops:
            raise RuntimeError(f"Flux planner: duplicate mapping for {dst} (src={src})")
        out_ops[dst] = ("copy", (_require(src),))
        consumed.add(src)

    def _add_concat(dst: str, srcs: Sequence[str]) -> None:
        if dst in out_ops:
            raise RuntimeError(f"Flux planner: duplicate mapping for {dst}")
        resolved = tuple(_require(s) for s in srcs)
        out_ops[dst] = ("concat_dim0", resolved)
        consumed.update(resolved)

    def _add_swap_halves(dst: str, src: str) -> None:
        """Swap first and second halves of tensor along dim0.

        Diffusers stores shift/scale chunks in opposite order to BFL/ComfyUI for
        final_layer.adaLN_modulation. This op corrects the layout during conversion.
        """
        if dst in out_ops:
            raise RuntimeError(f"Flux planner: duplicate mapping for {dst} (src={src})")
        out_ops[dst] = ("swap_halves", (_require(src),))
        consumed.add(src)

    # Top-level modules (input/output and embedder MLPs).
    _add_copy("img_in.weight", "x_embedder.weight")
    _add_copy("img_in.bias", "x_embedder.bias")
    _add_copy("txt_in.weight", "context_embedder.weight")
    _add_copy("txt_in.bias", "context_embedder.bias")

    _add_copy("final_layer.linear.weight", "proj_out.weight")
    _add_copy("final_layer.linear.bias", "proj_out.bias")
    # Diffusers stores shift/scale in [scale, shift] order but BFL/ComfyUI expects [shift, scale].
    _add_swap_halves("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight")
    _add_swap_halves("final_layer.adaLN_modulation.1.bias", "norm_out.linear.bias")

    _add_copy("time_in.in_layer.weight", "time_text_embed.timestep_embedder.linear_1.weight")
    _add_copy("time_in.in_layer.bias", "time_text_embed.timestep_embedder.linear_1.bias")
    _add_copy("time_in.out_layer.weight", "time_text_embed.timestep_embedder.linear_2.weight")
    _add_copy("time_in.out_layer.bias", "time_text_embed.timestep_embedder.linear_2.bias")

    _add_copy("vector_in.in_layer.weight", "time_text_embed.text_embedder.linear_1.weight")
    _add_copy("vector_in.in_layer.bias", "time_text_embed.text_embedder.linear_1.bias")
    _add_copy("vector_in.out_layer.weight", "time_text_embed.text_embedder.linear_2.weight")
    _add_copy("vector_in.out_layer.bias", "time_text_embed.text_embedder.linear_2.bias")

    # Guidance embedder is optional for schnell-style configs.
    if "time_text_embed.guidance_embedder.linear_1.weight" in keys_set:
        _add_copy("guidance_in.in_layer.weight", "time_text_embed.guidance_embedder.linear_1.weight")
        _add_copy("guidance_in.in_layer.bias", "time_text_embed.guidance_embedder.linear_1.bias")
        _add_copy("guidance_in.out_layer.weight", "time_text_embed.guidance_embedder.linear_2.weight")
        _add_copy("guidance_in.out_layer.bias", "time_text_embed.guidance_embedder.linear_2.bias")

    # Double blocks: Diffusers `transformer_blocks.N.*` → Comfy `double_blocks.N.*`.
    double_indices = _extract_block_indices(tensor_names, "transformer_blocks.")
    if not double_indices:
        raise RuntimeError("Flux planner: no transformer_blocks.* tensors found")
    if double_indices != list(range(0, max(double_indices) + 1)):
        raise RuntimeError(f"Flux planner: unexpected transformer_blocks indices: {double_indices}")

    for i in double_indices:
        base = f"transformer_blocks.{i}."

        def src(suffix: str) -> str:
            return base + suffix

        # Norm scales (weight → scale).
        _add_copy(f"double_blocks.{i}.img_attn.norm.query_norm.scale", src("attn.norm_q.weight"))
        _add_copy(f"double_blocks.{i}.img_attn.norm.key_norm.scale", src("attn.norm_k.weight"))
        _add_copy(f"double_blocks.{i}.txt_attn.norm.query_norm.scale", src("attn.norm_added_q.weight"))
        _add_copy(f"double_blocks.{i}.txt_attn.norm.key_norm.scale", src("attn.norm_added_k.weight"))

        # QKV projections.
        _add_concat(
            f"double_blocks.{i}.img_attn.qkv.weight",
            (src("attn.to_q.weight"), src("attn.to_k.weight"), src("attn.to_v.weight")),
        )
        _add_concat(
            f"double_blocks.{i}.img_attn.qkv.bias",
            (src("attn.to_q.bias"), src("attn.to_k.bias"), src("attn.to_v.bias")),
        )
        _add_concat(
            f"double_blocks.{i}.txt_attn.qkv.weight",
            (src("attn.add_q_proj.weight"), src("attn.add_k_proj.weight"), src("attn.add_v_proj.weight")),
        )
        _add_concat(
            f"double_blocks.{i}.txt_attn.qkv.bias",
            (src("attn.add_q_proj.bias"), src("attn.add_k_proj.bias"), src("attn.add_v_proj.bias")),
        )

        # Output projections.
        _add_copy(f"double_blocks.{i}.img_attn.proj.weight", src("attn.to_out.0.weight"))
        _add_copy(f"double_blocks.{i}.img_attn.proj.bias", src("attn.to_out.0.bias"))
        _add_copy(f"double_blocks.{i}.txt_attn.proj.weight", src("attn.to_add_out.weight"))
        _add_copy(f"double_blocks.{i}.txt_attn.proj.bias", src("attn.to_add_out.bias"))

        # Feed-forward MLPs.
        _add_copy(f"double_blocks.{i}.img_mlp.0.weight", src("ff.net.0.proj.weight"))
        _add_copy(f"double_blocks.{i}.img_mlp.0.bias", src("ff.net.0.proj.bias"))
        _add_copy(f"double_blocks.{i}.img_mlp.2.weight", src("ff.net.2.weight"))
        _add_copy(f"double_blocks.{i}.img_mlp.2.bias", src("ff.net.2.bias"))

        _add_copy(f"double_blocks.{i}.txt_mlp.0.weight", src("ff_context.net.0.proj.weight"))
        _add_copy(f"double_blocks.{i}.txt_mlp.0.bias", src("ff_context.net.0.proj.bias"))
        _add_copy(f"double_blocks.{i}.txt_mlp.2.weight", src("ff_context.net.2.weight"))
        _add_copy(f"double_blocks.{i}.txt_mlp.2.bias", src("ff_context.net.2.bias"))

        # Modulation.
        _add_copy(f"double_blocks.{i}.img_mod.lin.weight", src("norm1.linear.weight"))
        _add_copy(f"double_blocks.{i}.img_mod.lin.bias", src("norm1.linear.bias"))
        _add_copy(f"double_blocks.{i}.txt_mod.lin.weight", src("norm1_context.linear.weight"))
        _add_copy(f"double_blocks.{i}.txt_mod.lin.bias", src("norm1_context.linear.bias"))

    # Single blocks: Diffusers `single_transformer_blocks.N.*` → Comfy `single_blocks.N.*`.
    single_indices = _extract_block_indices(tensor_names, "single_transformer_blocks.")
    if not single_indices:
        raise RuntimeError("Flux planner: no single_transformer_blocks.* tensors found")
    if single_indices != list(range(0, max(single_indices) + 1)):
        raise RuntimeError(f"Flux planner: unexpected single_transformer_blocks indices: {single_indices}")

    for i in single_indices:
        base = f"single_transformer_blocks.{i}."

        def src(suffix: str) -> str:
            return base + suffix

        _add_copy(f"single_blocks.{i}.norm.query_norm.scale", src("attn.norm_q.weight"))
        _add_copy(f"single_blocks.{i}.norm.key_norm.scale", src("attn.norm_k.weight"))

        # Fused linear1: qkv + mlp projection.
        _add_concat(
            f"single_blocks.{i}.linear1.weight",
            (
                src("attn.to_q.weight"),
                src("attn.to_k.weight"),
                src("attn.to_v.weight"),
                src("proj_mlp.weight"),
            ),
        )
        _add_concat(
            f"single_blocks.{i}.linear1.bias",
            (
                src("attn.to_q.bias"),
                src("attn.to_k.bias"),
                src("attn.to_v.bias"),
                src("proj_mlp.bias"),
            ),
        )

        _add_copy(f"single_blocks.{i}.linear2.weight", src("proj_out.weight"))
        _add_copy(f"single_blocks.{i}.linear2.bias", src("proj_out.bias"))

        _add_copy(f"single_blocks.{i}.modulation.lin.weight", src("norm.linear.weight"))
        _add_copy(f"single_blocks.{i}.modulation.lin.bias", src("norm.linear.bias"))

    leftovers = sorted(keys_set.difference(consumed))
    if leftovers:
        sample = ", ".join(leftovers[:12])
        more = "" if len(leftovers) <= 12 else f" (+{len(leftovers) - 12} more)"
        raise RuntimeError(f"Flux planner: unmapped tensors: {sample}{more}")

    plans: list[TensorPlan] = []
    key_mapping: dict[str, str] = {}

    for gguf_name in sorted(out_ops):
        op, srcs = out_ops[gguf_name]
        if not srcs:
            raise RuntimeError(f"Flux planner: empty source list for {gguf_name}")

        if op == "copy":
            raw_shape = _shape_of(safetensors_handle, shapes, srcs[0])
            src_name = srcs[0]
        elif op == "swap_halves":
            # Shape stays the same; just reorder rows.
            raw_shape = _shape_of(safetensors_handle, shapes, srcs[0])
            src_name = srcs[0]
        elif op == "concat_dim0":
            base_shape = _shape_of(safetensors_handle, shapes, srcs[0])
            dims = len(base_shape)
            if dims not in (1, 2):
                raise RuntimeError(f"Flux planner: unexpected concat_dim0 source shape for {gguf_name}: {base_shape}")
            total0 = 0
            trailing = base_shape[1:] if dims == 2 else ()
            for src in srcs:
                shape = _shape_of(safetensors_handle, shapes, src)
                if len(shape) != dims:
                    raise RuntimeError(f"Flux planner: concat_dim0 rank mismatch for {gguf_name}: {src}={shape}")
                if dims == 2 and shape[1:] != trailing:
                    raise RuntimeError(
                        f"Flux planner: concat_dim0 trailing dims mismatch for {gguf_name}: {src}={shape} expected *x{trailing[0]}"
                    )
                total0 += int(shape[0])
            raw_shape = (total0, *trailing) if dims == 2 else (total0,)
            src_name = srcs[0]
        else:
            raise RuntimeError(f"Flux planner: unknown op={op!r} for {gguf_name}")

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

        if src_name not in key_mapping:
            key_mapping[src_name] = gguf_name

    return plans, key_mapping


__all__ = [
    "TensorPlan",
    "is_flux_transformer_config",
    "is_wan22_transformer_config",
    "is_zimage_transformer_config",
    "normalize_flux_transformer_metadata_config",
    "normalize_wan22_transformer_metadata_config",
    "normalize_zimage_transformer_metadata_config",
    "plan_tensors",
    "plan_flux_transformer_tensors",
    "plan_wan22_transformer_tensors",
    "plan_zimage_transformer_tensors",
]
