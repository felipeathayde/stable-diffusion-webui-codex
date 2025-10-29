from __future__ import annotations

from typing import Dict

import torch

from apps.backend.runtime.model_registry.specs import ModelSignature

from ..builders import build_estimated_config, register_text_encoder
from ..converters.clip import convert_sd20_clip
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
            SplitSpec(name="vae", prefixes=("first_stage_model.",), required=False),
            SplitSpec(name="text_encoder", prefixes=("conditioner.embedders.0.model.", "cond_stage_model.model.")),
        ],
        converters=(
            ConverterSpec(component="text_encoder", function=_convert_clip),
        ),
        validations=(
            ValidationSpec(name="unet_channels", function=_validate_unet_channels),
            ValidationSpec(name="clip_presence", function=_validate_clip_keys),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))


def _convert_clip(tensors: Dict[str, torch.Tensor], context):
    converted = convert_sd20_clip(tensors)
    register_text_encoder(context, "clip_h", "text_encoder")
    return converted


def _validate_unet_channels(context):
    unet = context.require("unet").tensors
    weight = unet.get("input_blocks.0.0.weight")
    if not isinstance(weight, torch.Tensor):
        raise ValidationError("Expected 'input_blocks.0.0.weight' in UNet state dict", component="unet")
    expected = context.signature.core.channels_in
    if weight.shape[1] != expected:
        raise ValidationError(
            f"UNet channels_in mismatch: expected {expected}, found {weight.shape[1]}",
            component="unet",
        )


def _validate_clip_keys(context):
    clip = context.require("text_encoder").tensors
    required = "transformer.text_model.encoder.layers.0.layer_norm1.weight"
    if required not in clip:
        raise ValidationError(f"Missing key '{required}' after conversion", component="text_encoder")
