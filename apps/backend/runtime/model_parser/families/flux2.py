"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: FLUX.2 parser plan builder for core-only Klein 4B/base-4B checkpoints.
Builds a strict parser plan for the supported FLUX.2 Comfy/Codex core-only SafeTensors layout, converts the transformer weights into the
Diffusers `Flux2Transformer2DModel` keyspace, and registers a single external Qwen3-4B text-encoder alias for override resolution.
Unsupported layouts/configs (Diffusers exports, embedded assets, non-4B variants) fail loud.

Symbols (top-level; keep in sync; no ghosts):
- `_FLUX2_REQUIRED_KEYS` (constant): Required raw FLUX.2 core keys before conversion.
- `_FLUX2_SUPPORTED_PREFIXES` (constant): Supported wrapper prefixes for core-only FLUX.2 transformer checkpoints.
- `_register_flux2_text_encoders` (function): Registers the `qwen3_4b` alias mapping in the parser context.
- `build_plan` (function): Builds and returns the FLUX.2 `ParserPlanBundle`.
- `_convert_flux2_transformer` (function): Converts the raw FLUX.2 transformer checkpoint into Diffusers keyspace.
- `_validate_flux2_transformer_component` (function): Validates the converted FLUX.2 transformer component.
- `_assert_supported_flux2_core_layout` (function): Validates the supported FLUX.2 4B/base-4B raw layout contract.
- `_convert_ada_layer_norm_weights` (function): Swaps FLUX.2 adaLN scale/shift weights into Diffusers order.
- `_convert_flux2_double_stream_blocks` (function): Converts `double_blocks.*` tensors into Diffusers `transformer_blocks.*` keys.
- `_convert_flux2_single_stream_blocks` (function): Converts `single_blocks.*` tensors into Diffusers `single_transformer_blocks.*` keys.
- `_rename_in_place` (function): Renames one state-dict key in place.
- `_shape_2d` (function): Returns the 2D tensor shape for a required key.
- `_split_qkv` (function): Splits fused QKV tensors into three chunks.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch

from apps.backend.runtime.model_registry.specs import ModelSignature

from ..builders import build_estimated_config, register_text_encoder
from ..errors import ValidationError
from ..quantization import validate_component_dtypes
from ..specs import (
    ConverterSpec,
    ParserPlan,
    ParserPlanBundle,
    SplitSpec,
    ValidationSpec,
)

_FLUX2_REQUIRED_KEYS = (
    "img_in.weight",
    "txt_in.weight",
    "time_in.in_layer.weight",
    "time_in.out_layer.weight",
    "double_stream_modulation_img.lin.weight",
    "double_stream_modulation_txt.lin.weight",
    "single_stream_modulation.lin.weight",
    "double_blocks.0.img_attn.qkv.weight",
    "double_blocks.0.txt_attn.qkv.weight",
    "double_blocks.0.img_attn.norm.key_norm.scale",
    "double_blocks.0.txt_attn.norm.key_norm.scale",
    "single_blocks.0.linear1.weight",
    "single_blocks.0.linear2.weight",
    "single_blocks.0.norm.key_norm.scale",
    "final_layer.linear.weight",
    "final_layer.adaLN_modulation.1.weight",
)

_FLUX2_SUPPORTED_PREFIXES = (
    "transformer.",
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
    "",
)

_FLUX2_SIMPLE_RENAMES = (
    ("img_in", "x_embedder"),
    ("txt_in", "context_embedder"),
    ("time_in.in_layer", "time_guidance_embed.timestep_embedder.linear_1"),
    ("time_in.out_layer", "time_guidance_embed.timestep_embedder.linear_2"),
    ("double_stream_modulation_img.lin", "double_stream_modulation_img.linear"),
    ("double_stream_modulation_txt.lin", "double_stream_modulation_txt.linear"),
    ("single_stream_modulation.lin", "single_stream_modulation.linear"),
    ("final_layer.linear", "proj_out"),
)

_FLUX2_DOUBLE_BLOCK_KEY_MAP = {
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.norm.key_norm": "attn.norm_k",
    "img_attn.proj": "attn.to_out.0",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_attn.norm.query_norm": "attn.norm_added_q",
    "txt_attn.norm.key_norm": "attn.norm_added_k",
    "txt_attn.proj": "attn.to_add_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}

_FLUX2_SINGLE_BLOCK_KEY_MAP = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}


def _register_flux2_text_encoders(context) -> None:
    register_text_encoder(context, "qwen3_4b", "text_encoder")


def build_plan(signature: ModelSignature) -> ParserPlanBundle:
    plan = ParserPlan(
        splits=[
            SplitSpec(name="transformer", prefixes=_FLUX2_SUPPORTED_PREFIXES),
        ],
        converters=(
            ConverterSpec(component="transformer", function=_convert_flux2_transformer),
        ),
        validations=(
            ValidationSpec(name="register_flux2_text_encoders", function=_register_flux2_text_encoders),
            ValidationSpec(name="flux2_transformer", function=_validate_flux2_transformer_component),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))


def _convert_flux2_transformer(tensors: Dict[str, Any], context) -> Dict[str, Any]:
    _assert_supported_flux2_core_layout(tensors, signature=context.signature)

    converted = dict(tensors)
    for old, new in _FLUX2_SIMPLE_RENAMES:
        for key in list(converted.keys()):
            if old not in key:
                continue
            _rename_in_place(converted, key, key.replace(old, new))

    for key in list(converted.keys()):
        if "final_layer.adaLN_modulation.1" in key:
            _convert_ada_layer_norm_weights(key, converted)
    for key in list(converted.keys()):
        if "double_blocks." in key:
            _convert_flux2_double_stream_blocks(key, converted)
    for key in list(converted.keys()):
        if "single_blocks." in key:
            _convert_flux2_single_stream_blocks(key, converted)

    return converted


def _validate_flux2_transformer_component(context) -> None:
    transformer = context.require("transformer").tensors
    required = (
        "x_embedder.weight",
        "context_embedder.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight",
        "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.linear.weight",
        "single_stream_modulation.linear.weight",
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.add_q_proj.weight",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
        "proj_out.weight",
        "norm_out.linear.weight",
    )
    missing = [key for key in required if key not in transformer]
    if missing:
        raise ValidationError(
            "FLUX.2 transformer conversion is incomplete; missing Diffusers keys. "
            f"missing_sample={missing[:10]}",
            component="transformer",
        )


def _assert_supported_flux2_core_layout(tensors: Mapping[str, Any], *, signature: ModelSignature) -> None:
    missing = [key for key in _FLUX2_REQUIRED_KEYS if key not in tensors]
    if missing:
        if any(key.startswith(("x_embedder.", "transformer_blocks.", "single_transformer_blocks.")) for key in tensors):
            raise ValidationError(
                "FLUX.2 transformer looks like a Diffusers export; expected the core-only Comfy/Codex layout "
                "(`double_blocks.*`, `single_blocks.*`, `img_in.*`, `txt_in.*`).",
                component="transformer",
            )
        raise ValidationError(
            "FLUX.2 transformer is missing required core-only keys. "
            f"missing_sample={missing[:10]}",
            component="transformer",
        )

    unsupported_prefixes = ("text_encoder.", "text_encoders.", "vae.", "guidance_in.", "vector_in.")
    embedded = [key for key in tensors if key.startswith(unsupported_prefixes)]
    if embedded:
        raise ValidationError(
            "FLUX.2 core-only slice does not support embedded text encoder/VAE/guidance assets. "
            f"embedded_sample={embedded[:10]}",
            component="transformer",
        )

    expected_in_channels = int(signature.core.channels_in)
    expected_context_dim = int(signature.core.context_dim or 0)
    expected_double = int((signature.extras or {}).get("flow_double_layers", 0))
    expected_single = int((signature.extras or {}).get("flow_single_layers", 0))

    img_in_shape = _shape_2d(tensors, "img_in.weight", component="transformer")
    txt_in_shape = _shape_2d(tensors, "txt_in.weight", component="transformer")
    final_shape = _shape_2d(tensors, "final_layer.linear.weight", component="transformer")

    hidden_dim = int(img_in_shape[0])
    if int(img_in_shape[1]) != expected_in_channels:
        raise ValidationError(
            "FLUX.2 image input projection channel mismatch. "
            f"got={img_in_shape[1]} expected={expected_in_channels}",
            component="transformer",
        )
    if int(txt_in_shape[0]) != hidden_dim:
        raise ValidationError(
            "FLUX.2 hidden-dim mismatch between img_in and txt_in projections. "
            f"img_hidden={hidden_dim} txt_hidden={txt_in_shape[0]}",
            component="transformer",
        )
    if int(txt_in_shape[1]) != expected_context_dim:
        raise ValidationError(
            "Unsupported FLUX.2 context dimension. Only Klein 4B/base-4B is supported. "
            f"got={txt_in_shape[1]} expected={expected_context_dim}",
            component="transformer",
        )
    if int(final_shape[0]) != expected_in_channels or int(final_shape[1]) != hidden_dim:
        raise ValidationError(
            "FLUX.2 final projection shape mismatch. "
            f"got={final_shape} expected=({expected_in_channels}, {hidden_dim})",
            component="transformer",
        )

    double_layers = sum(1 for idx in range(expected_double) if any(k.startswith(f"double_blocks.{idx}.") for k in tensors))
    if double_layers != expected_double or any(k.startswith(f"double_blocks.{expected_double}.") for k in tensors):
        raise ValidationError(
            "Unsupported FLUX.2 double-block depth. Only Klein 4B/base-4B is supported. "
            f"got={double_layers} expected={expected_double}",
            component="transformer",
        )

    single_layers = sum(1 for idx in range(expected_single) if any(k.startswith(f"single_blocks.{idx}.") for k in tensors))
    if single_layers != expected_single or any(k.startswith(f"single_blocks.{expected_single}.") for k in tensors):
        raise ValidationError(
            "Unsupported FLUX.2 single-block depth. Only Klein 4B/base-4B is supported. "
            f"got={single_layers} expected={expected_single}",
            component="transformer",
        )


def _convert_ada_layer_norm_weights(key: str, state_dict: dict[str, Any]) -> None:
    if not key.endswith(".weight"):
        raise ValidationError(
            f"Unsupported FLUX.2 adaLN parameter {key!r}; expected weight-only adaLN tensor.",
            component="transformer",
        )
    new_key = key.replace("final_layer.adaLN_modulation.1", "norm_out.linear")
    shift, scale = _split_qkv(state_dict.pop(key), context=key, chunks=2)
    state_dict[new_key] = torch.cat((scale, shift), dim=0)


def _convert_flux2_double_stream_blocks(key: str, state_dict: dict[str, Any]) -> None:
    if not (key.endswith(".weight") or key.endswith(".bias") or key.endswith(".scale")):
        return
    if "double_blocks." not in key:
        return

    parts = key.split(".")
    if len(parts) < 5:
        raise ValidationError(f"Malformed FLUX.2 double-block key: {key!r}", component="transformer")

    block_idx = parts[1]
    within_block = ".".join(parts[2:-1])
    param_type = parts[-1]
    if param_type == "scale":
        param_type = "weight"

    if within_block.endswith("qkv"):
        fused = state_dict.pop(key)
        q, k, v = _split_qkv(fused, context=key)
        if within_block.startswith("img_attn"):
            names = ("attn.to_q", "attn.to_k", "attn.to_v")
        elif within_block.startswith("txt_attn"):
            names = ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")
        else:
            raise ValidationError(f"Unsupported FLUX.2 fused-QKV block: {key!r}", component="transformer")
        for name, tensor in zip(names, (q, k, v), strict=True):
            state_dict[f"transformer_blocks.{block_idx}.{name}.{param_type}"] = tensor
        return

    target = _FLUX2_DOUBLE_BLOCK_KEY_MAP.get(within_block)
    if target is None:
        raise ValidationError(f"Unsupported FLUX.2 double-block tensor: {key!r}", component="transformer")
    _rename_in_place(state_dict, key, f"transformer_blocks.{block_idx}.{target}.{param_type}")


def _convert_flux2_single_stream_blocks(key: str, state_dict: dict[str, Any]) -> None:
    if not (key.endswith(".weight") or key.endswith(".bias") or key.endswith(".scale")):
        return
    if "single_blocks." not in key:
        return

    parts = key.split(".")
    if len(parts) < 5:
        raise ValidationError(f"Malformed FLUX.2 single-block key: {key!r}", component="transformer")

    block_idx = parts[1]
    within_block = ".".join(parts[2:-1])
    param_type = parts[-1]
    if param_type == "scale":
        param_type = "weight"

    target = _FLUX2_SINGLE_BLOCK_KEY_MAP.get(within_block)
    if target is None:
        raise ValidationError(f"Unsupported FLUX.2 single-block tensor: {key!r}", component="transformer")
    _rename_in_place(state_dict, key, f"single_transformer_blocks.{block_idx}.{target}.{param_type}")


def _rename_in_place(state_dict: dict[str, Any], old_key: str, new_key: str) -> None:
    if old_key == new_key:
        return
    state_dict[new_key] = state_dict.pop(old_key)


def _shape_2d(tensors: Mapping[str, Any], key: str, *, component: str) -> tuple[int, int]:
    tensor = tensors.get(key)
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValidationError(
            f"FLUX.2 tensor {key!r} must be rank-2, got shape={shape!r}",
            component=component,
        )
    return int(shape[0]), int(shape[1])


def _split_qkv(tensor: Any, *, context: str, chunks: int = 3) -> tuple[torch.Tensor, ...]:
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(
            f"FLUX.2 tensor {context!r} must be a torch.Tensor, got {type(tensor).__name__}",
            component="transformer",
        )
    if tensor.shape[0] % chunks != 0:
        raise ValidationError(
            f"FLUX.2 tensor {context!r} first dimension must be divisible by {chunks}, got {tuple(tensor.shape)!r}",
            component="transformer",
        )
    return tuple(torch.chunk(tensor, chunks, dim=0))
