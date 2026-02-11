"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SD3/SD3.5 parser plan builder (DiT core + optional VAE + CLIP-L/CLIP-G/T5XXL).
Defines split/conversion/validation steps for SD3-family checkpoints, converting embedded CLIP and (optional) T5-XXL components and
validating core presence and required text-encoder keys. CLIP-G conversion keeps native projection orientation in `auto` mode.

Symbols (top-level; keep in sync; no ghosts):
- `build_plan` (function): Builds and returns the SD3 `ParserPlanBundle`.
- `_convert_clip_l` (function): Converts CLIP-L tensors and registers the `clip_l` alias mapping.
- `_convert_clip_g` (function): Converts CLIP-G tensors when present (auto/native projection orientation) and registers the `clip_g` alias mapping.
- `_convert_t5` (function): Converts T5-XXL tensors when present and registers the `t5xxl` alias mapping.
- `_validate_transformer_core` (function): Validates presence of required SD3 transformer keys.
- `_validate_clip_l` (function): Validates CLIP-L conversion output presence.
- `_validate_clip_g` (function): Validates CLIP-G conversion output presence when provided.
- `_validate_t5` (function): Validates T5 conversion output keys when provided.
"""

from __future__ import annotations

from typing import Dict

import torch

from apps.backend.runtime.model_registry.specs import ModelSignature

from ..builders import build_estimated_config, register_text_encoder
from ..converters import convert_clip, convert_t5xxl_encoder
from ..errors import ValidationError
from ..specs import (
    ParserPlan,
    ParserPlanBundle,
    SplitSpec,
    ConverterSpec,
    ValidationSpec,
)
from ..quantization import validate_component_dtypes


def build_plan(signature: ModelSignature) -> ParserPlanBundle:
    plan = ParserPlan(
        splits=[
            SplitSpec(name="transformer", prefixes=("model.diffusion_model.",)),
            SplitSpec(name="vae", prefixes=("vae.",), required=False),
            SplitSpec(name="text_encoder", prefixes=("text_encoders.clip_l.",)),
            SplitSpec(name="text_encoder_2", prefixes=("text_encoders.clip_g.",), required=False),
            SplitSpec(name="text_encoder_3", prefixes=("text_encoders.t5xxl.",), required=False),
        ],
        converters=(
            ConverterSpec(component="text_encoder", function=_convert_clip_l),
            ConverterSpec(component="text_encoder_2", function=_convert_clip_g),
            ConverterSpec(component="text_encoder_3", function=_convert_t5),
        ),
        validations=(
            ValidationSpec(name="core_presence", function=_validate_transformer_core),
            ValidationSpec(name="clip_l_presence", function=_validate_clip_l),
            ValidationSpec(name="clip_g_presence", function=_validate_clip_g),
            ValidationSpec(name="t5_presence", function=_validate_t5),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))


def _convert_clip_l(tensors: Dict[str, torch.Tensor], context):
    converted = convert_clip(
        tensors,
        alias="clip_l",
        layers=32,
        ensure_position_ids=True,
        drop_logit_scale=True,
    )
    register_text_encoder(context, "clip_l", "text_encoder")
    return converted


def _convert_clip_g(tensors: Dict[str, torch.Tensor], context):
    if tensors:
        converted = convert_clip(
            tensors,
            alias="clip_g",
            layers=32,
            ensure_position_ids=True,
            drop_logit_scale=True,
            projection_orientation="auto",
        )
        register_text_encoder(context, "clip_g", "text_encoder_2")
        return converted
    return tensors


def _convert_t5(tensors: Dict[str, torch.Tensor], context):
    if tensors:
        converted = convert_t5xxl_encoder(tensors)
        register_text_encoder(context, "t5xxl", "text_encoder_3")
        return converted
    return tensors


def _validate_transformer_core(context):
    unet = context.require("transformer").tensors
    key = "joint_blocks.0.context_block.attn.qkv.weight"
    if key not in unet:
        raise ValidationError(f"Missing key '{key}' in SD3 transformer", component="transformer")


def _validate_clip_l(context):
    clip = context.components.get("text_encoder")
    if clip is None or "transformer.text_model.embeddings.token_embedding.weight" not in clip.tensors:
        raise ValidationError("SD3 CLIP-L conversion failed", component="text_encoder")


def _validate_clip_g(context):
    clip = context.components.get("text_encoder_2")
    if clip and "transformer.text_model.embeddings.token_embedding.weight" not in clip.tensors:
        raise ValidationError("SD3 CLIP-G conversion failed", component="text_encoder_2")


def _validate_t5(context):
    t5 = context.components.get("text_encoder_3")
    if t5:
        key = "transformer.encoder.final_layer_norm.weight"
        if key not in t5.tensors:
            raise ValidationError("SD3 T5XXL conversion missing final layer norm", component="text_encoder_3")
