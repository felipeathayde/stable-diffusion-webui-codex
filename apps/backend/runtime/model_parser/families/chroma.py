from __future__ import annotations

from typing import Dict

import torch

from apps.backend.runtime.model_registry.specs import ModelSignature

from ..builders import build_estimated_config, register_text_encoder
from ..converters import convert_t5xxl_encoder
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
            SplitSpec(name="transformer", prefixes=("transformer.", "model.diffusion_model.")),
            SplitSpec(name="vae", prefixes=("vae.",), required=False),
            SplitSpec(name="text_encoder", prefixes=("text_encoder.",)),
        ],
        converters=(
            ConverterSpec(component="text_encoder", function=_convert_t5),
        ),
        validations=(
            ValidationSpec(name="core_presence", function=_validate_transformer_core),
            ValidationSpec(name="t5_presence", function=_validate_t5),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))


def _convert_t5(tensors: Dict[str, torch.Tensor], context):
    converted = convert_t5xxl_encoder(tensors)
    register_text_encoder(context, "t5xxl", "text_encoder")
    return converted


def _validate_transformer_core(context):
    unet = context.require("transformer").tensors
    key = "double_blocks.0.img_attn.norm.key_norm.scale"
    if key not in unet:
        raise ValidationError("Chroma transformer missing double block attn scale", component="transformer")


def _validate_t5(context):
    t5 = context.require("text_encoder").tensors
    key = "transformer.encoder.final_layer_norm.weight"
    if key not in t5:
        raise ValidationError("Chroma T5 conversion missing final layer norm", component="text_encoder")
