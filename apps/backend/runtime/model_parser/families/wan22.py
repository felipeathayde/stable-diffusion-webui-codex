"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 parser plan builder (temporal DiT core + optional VAE + text encoders).
Builds split/conversion/validation steps for WAN22 checkpoints using signature-provided core prefixes and text-encoder signatures, converting
UMT5-XXL (required) and CLIP-L when present and validating core patch/head weights.

Symbols (top-level; keep in sync; no ghosts):
- `_REQUIRED_ENCODERS` (constant): Set of required encoder names (currently `umt5xxl`).
- `_CLIP_LAYERS` (constant): CLIP-L layer count used by WAN22 conversions.
- `build_plan` (function): Builds and returns the WAN22 `ParserPlanBundle`.
- `_component_name` (function): Derives a component name for a `TextEncoderSignature` (`text_encoder_<name>`).
- `_converter_for` (function): Returns a converter closure for a given text encoder signature (registers alias mapping).
- `_validator_for` (function): Returns a validator closure for a given text encoder signature.
- `_validate_core` (function): Validates presence of required WAN transformer keys.
- `_has_any` (function): Checks whether any key ends with one of the given suffixes.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch

from apps.backend.runtime.model_registry.specs import ModelSignature, TextEncoderSignature

from ..builders import build_estimated_config, register_text_encoder
from ..converters import convert_clip, convert_umt5_encoder
from ..errors import ValidationError
from ..specs import (
    ConverterSpec,
    ParserPlan,
    ParserPlanBundle,
    SplitSpec,
    ValidationSpec,
)
from ..quantization import validate_component_dtypes

_REQUIRED_ENCODERS = {"umt5xxl"}
_CLIP_LAYERS = 32


def build_plan(signature: ModelSignature) -> ParserPlanBundle:
    core_prefixes = tuple(signature.core.key_prefixes) if signature.core.key_prefixes else ("model.diffusion_model.",)
    splits = [SplitSpec(name="transformer", prefixes=core_prefixes)]
    splits.append(SplitSpec(name="vae", prefixes=("vae.",), required=False))

    converters = []
    validations = [ValidationSpec(name="wan_core_presence", function=_validate_core)]

    for te in signature.text_encoders:
        component_name = _component_name(te)
        splits.append(
            SplitSpec(
                name=component_name,
                prefixes=(te.key_prefix,),
                required=te.name in _REQUIRED_ENCODERS,
            )
        )
        converters.append(ConverterSpec(component=component_name, function=_converter_for(te)))
        validations.append(ValidationSpec(name=f"wan_validate_{te.name}", function=_validator_for(te)))

    validations.append(ValidationSpec(name="dtype_sanity", function=validate_component_dtypes))

    plan = ParserPlan(splits=splits, converters=tuple(converters), validations=tuple(validations))
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))


def _component_name(te: TextEncoderSignature) -> str:
    return f"text_encoder_{te.name}"


def _converter_for(te: TextEncoderSignature):
    component = _component_name(te)

    if te.name == "umt5xxl":
        def _convert(tensors: Dict[str, torch.Tensor], context):
            if not tensors:
                raise ValidationError("Wan2.2 umt5xxl encoder missing", component=component)
            converted = convert_umt5_encoder(tensors)
            register_text_encoder(context, "umt5xxl", component)
            return converted
        return _convert

    if te.name == "clip_l":
        def _convert(tensors: Dict[str, torch.Tensor], context):
            if not tensors:
                raise ValidationError("Wan2.2 CLIP-L encoder missing", component=component)
            converted = convert_clip(
                tensors,
                alias="clip_l",
                layers=_CLIP_LAYERS,
                ensure_position_ids=True,
                drop_logit_scale=True,
            )
            register_text_encoder(context, "clip_l", component)
            return converted
        return _convert

    # For any additional text encoders we default to pass-through with registration.
    def _convert_passthrough(tensors: Dict[str, torch.Tensor], context):
        register_text_encoder(context, te.name, component)
        return tensors

    return _convert_passthrough


def _validator_for(te: TextEncoderSignature):
    component = _component_name(te)

    if te.name == "umt5xxl":
        expected_key = "transformer.encoder.final_layer_norm.weight"
    elif te.name.startswith("clip"):
        expected_key = "transformer.text_model.embeddings.token_embedding.weight"
    else:
        expected_key = None

    def _validate(context):
        state = context.components.get(component)
        if te.name in _REQUIRED_ENCODERS and (state is None or not state.tensors):
            raise ValidationError(f"Wan2.2 required encoder '{te.name}' missing", component=component)
        if expected_key and state and expected_key not in state.tensors:
            raise ValidationError(
                f"Wan2.2 encoder '{te.name}' missing key '{expected_key}'",
                component=component,
            )

    return _validate


def _validate_core(context):
    unet = context.require("transformer").tensors
    if not _has_any(unet.keys(), ("patch_embedding.weight", "head.head.weight")):
        raise ValidationError("Wan2.2 transformer missing patch/head weights", component="transformer")


def _has_any(keys: Iterable[str], suffixes: Iterable[str]) -> bool:
    for key in keys:
        for suffix in suffixes:
            if key.endswith(suffix):
                return True
    return False
