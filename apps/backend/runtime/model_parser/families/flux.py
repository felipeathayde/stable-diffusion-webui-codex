from __future__ import annotations

from typing import Dict

import torch

from apps.backend.runtime.model_registry.specs import ModelSignature, QuantizationKind

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


_FLUX_CORE_PREFIXES = ("transformer.", "model.diffusion_model.")


def build_plan(signature: ModelSignature) -> ParserPlanBundle:
    # GGUF core-only checkpoints: only the rectified-flow backbone lives in the
    # state_dict (double_blocks.+guidance), while CLIP/T5/VAE come from the
    # diffusers repo or external paths. For these, keep the plan minimal and
    # avoid text-encoder/vae validations.
    if signature.quantization.kind == QuantizationKind.GGUF:
        plan = ParserPlan(
            splits=[
                # Core-only: include all tensors under the single transformer component.
                SplitSpec(name="transformer", prefixes=("",)),
            ],
            converters=(),
            validations=(
                ValidationSpec(name="core_presence", function=_validate_transformer_core),
                ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
            ),
        )
        return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))

    # Full Flux checkpoints: expect transformer + VAE + both text encoders.
    plan = ParserPlan(
        splits=[
            SplitSpec(name="transformer", prefixes=_FLUX_CORE_PREFIXES),
            SplitSpec(name="vae", prefixes=("vae.",), required=False),
            SplitSpec(name="text_encoder", prefixes=("text_encoders.clip_l.",)),
            SplitSpec(name="text_encoder_2", prefixes=("text_encoders.t5xxl.",)),
        ],
        converters=(
            ConverterSpec(component="text_encoder", function=_convert_clip_l),
            ConverterSpec(component="text_encoder_2", function=_convert_t5),
        ),
        validations=(
            ValidationSpec(name="core_presence", function=_validate_transformer_core),
            ValidationSpec(name="clip_l_presence", function=_validate_clip_l),
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


def _convert_t5(tensors: Dict[str, torch.Tensor], context):
    converted = convert_t5xxl_encoder(tensors)
    register_text_encoder(context, "t5xxl", "text_encoder_2")
    return converted


def _validate_transformer_core(context):
    unet = context.require("transformer").tensors
    key = "double_blocks.0.img_attn.norm.key_norm.scale"
    if key not in unet:
        raise ValidationError("Flux transformer missing double block attn scale", component="transformer")


def _validate_clip_l(context):
    clip = context.require("text_encoder").tensors
    key = "transformer.text_model.encoder.layers.0.layer_norm1.weight"
    if key not in clip:
        raise ValidationError("Flux CLIP-L conversion failed", component="text_encoder")


def _validate_t5(context):
    t5 = context.require("text_encoder_2").tensors
    key = "transformer.encoder.final_layer_norm.weight"
    if key not in t5:
        raise ValidationError("Flux T5 conversion missing final layer norm", component="text_encoder_2")
