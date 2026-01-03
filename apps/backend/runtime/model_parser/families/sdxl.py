"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDXL parser plan builder (UNet + optional VAE + CLIP-L/CLIP-G).
Defines split/conversion/validation steps for SDXL base and refiner checkpoints, converting embedded CLIP encoders, normalizing label-embedding
keys for Forge-style variants, and validating required UNet/CLIP tensors.

Symbols (top-level; keep in sync; no ghosts):
- `build_plan` (function): Builds and returns the SDXL `ParserPlanBundle` (also used for SDXL refiner).
- `_CLIP_L_REQUIRED` (constant): Required CLIP-L keys checked after conversion.
- `_CLIP_G_REQUIRED` (constant): Required CLIP-G keys checked after conversion.
- `_convert_clip_l` (function): Converts CLIP-L tensors and registers the `clip_l` alias mapping.
- `_convert_clip_g` (function): Converts CLIP-G tensors and registers the `clip_g` alias mapping.
- `_validate_unet_channels` (function): Validates UNet `channels_in` vs the `ModelSignature` expectation.
- `_validate_clip_l` (function): Validates CLIP-L required keys exist after conversion.
- `_validate_clip_g` (function): Validates CLIP-G required keys exist after conversion.
- `_normalize_unet_label_embeddings` (function): Normalizes Forge-style nested SDXL label-embedding keys on the UNet component.
"""

from __future__ import annotations

from typing import Dict

import torch

from apps.backend.runtime.model_registry.specs import ModelSignature

from ..builders import build_estimated_config, register_text_encoder
from ..converters.clip import convert_sdxl_clip_g, convert_sdxl_clip_l
from ..converters.unet import normalize_label_embeddings
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
            SplitSpec(name="unet", prefixes=("model.diffusion_model.",)),
            SplitSpec(name="vae", prefixes=("first_stage_model.", "vae."), required=False),
            SplitSpec(name="text_encoder", prefixes=("conditioner.embedders.0.",)),
            SplitSpec(name="text_encoder_2", prefixes=("conditioner.embedders.1.model.",)),
        ],
        converters=(
            ConverterSpec(component="unet", function=_normalize_unet_label_embeddings),
            ConverterSpec(component="text_encoder", function=_convert_clip_l),
            ConverterSpec(component="text_encoder_2", function=_convert_clip_g),
        ),
        validations=(
            ValidationSpec(name="unet_channels", function=_validate_unet_channels),
            ValidationSpec(name="clip_l_presence", function=_validate_clip_l),
            ValidationSpec(name="clip_g_presence", function=_validate_clip_g),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))


_CLIP_L_REQUIRED = (
    "transformer.text_model.embeddings.token_embedding.weight",
    "transformer.text_model.encoder.layers.0.layer_norm1.weight",
    "transformer.text_model.final_layer_norm.weight",
)


_CLIP_G_REQUIRED = (
    "transformer.text_model.embeddings.token_embedding.weight",
    "transformer.text_model.encoder.layers.0.layer_norm1.weight",
    "transformer.text_model.final_layer_norm.weight",
)

# text_projection.weight is optional: IntegratedCLIP always creates the layer,
# and the loader tolerates missing projection weights (Forge-compatible).


def _convert_clip_l(tensors: Dict[str, torch.Tensor], context):
    converted = convert_sdxl_clip_l(tensors)
    register_text_encoder(context, "clip_l", "text_encoder")
    return converted


def _convert_clip_g(tensors: Dict[str, torch.Tensor], context):
    converted = convert_sdxl_clip_g(tensors)
    register_text_encoder(context, "clip_g", "text_encoder_2")
    return converted


def _validate_unet_channels(context):
    unet = context.require("unet").tensors
    key = "input_blocks.0.0.weight"
    weight = unet.get(key)
    if not isinstance(weight, torch.Tensor):
        raise ValidationError(f"Expected '{key}' in UNet state dict", component="unet")
    expected = context.signature.core.channels_in
    if weight.shape[1] != expected:
        raise ValidationError(
            f"UNet channels_in mismatch: expected {expected}, found {weight.shape[1]}",
            component="unet",
        )


def _validate_clip_l(context):
    clip = context.require("text_encoder").tensors
    missing = [key for key in _CLIP_L_REQUIRED if key not in clip]
    if missing:
        sample = ", ".join(missing[:3])
        raise ValidationError(
            f"SDXL CLIP-L is missing required tensors ({sample}); re-download the checkpoint or supply intact CLIP weights.",
            component="text_encoder",
        )


def _validate_clip_g(context):
    clip = context.require("text_encoder_2").tensors
    missing = [key for key in _CLIP_G_REQUIRED if key not in clip]
    if missing:
        sample = ", ".join(missing[:3])
        raise ValidationError(
            f"SDXL CLIP-G is missing required tensors ({sample}); re-download the checkpoint or supply intact CLIP weights.",
            component="text_encoder_2",
        )
def _normalize_unet_label_embeddings(tensors: Dict[str, torch.Tensor], context):
    return normalize_label_embeddings(tensors)
