from __future__ import annotations

from typing import Dict

import torch

from apps.backend.runtime.model_registry.specs import ModelSignature

from ..builders import build_estimated_config, register_text_encoder
from ..converters.clip import convert_sdxl_clip_g, convert_sdxl_clip_l
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
    key = "transformer.text_model.encoder.layers.0.layer_norm1.weight"
    if key not in clip:
        # Some SDXL variants ship pruned/partial CLIP-L encoders; allow loading and warn loudly.
        import logging
        logging.getLogger("backend.model_parser.sdxl").warning(
            "SDXL CLIP-L validation: missing %s; proceeding with partial encoder.", key
        )
        print(f"[parser][warn] SDXL CLIP-L missing '{key}'; continuing.", flush=True)


def _validate_clip_g(context):
    clip = context.require("text_encoder_2").tensors
    key = "transformer.text_model.encoder.layers.0.layer_norm1.weight"
    if key not in clip:
        import logging
        logging.getLogger("backend.model_parser.sdxl").warning(
            "SDXL CLIP-G validation: missing %s; proceeding with partial encoder.", key
        )
        print(f"[parser][warn] SDXL CLIP-G missing '{key}'; continuing.", flush=True)
