"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z Image parser plan builder (core-only DiT checkpoints).
Builds split/validation steps for Z Image core checkpoints, including GGUF core-only and prefixed SafeTensors exports, and registers a stable
Qwen3 text-encoder alias mapping for override resolution.

Symbols (top-level; keep in sync; no ghosts):
- `_ZIMAGE_CORE_PREFIXES` (constant): Accepted prefixes for Z Image core weights.
- `_register_zimage_text_encoders` (function): Registers the `qwen3_4b` alias mapping in the parser context.
- `build_plan` (function): Builds and returns the Z Image `ParserPlanBundle`.
- `_validate_transformer_core` (function): Validates presence of key transformer tensors for Z Image.
"""

from __future__ import annotations

from typing import Dict

import torch

from apps.backend.runtime.model_registry.specs import ModelSignature, QuantizationKind

from ..builders import build_estimated_config, register_text_encoder
from ..errors import ValidationError
from ..specs import (
    ParserPlan,
    ParserPlanBundle,
    SplitSpec,
    ConverterSpec,
    ValidationSpec,
)
from ..quantization import validate_component_dtypes


_ZIMAGE_CORE_PREFIXES = ("", "model.", "model.diffusion_model.")


def _register_zimage_text_encoders(context) -> None:
    """Register expected Z Image text encoder aliases even when weights are external.

    For GGUF core-only checkpoints, Qwen3 and VAE do not live in the primary
    state_dict. We still need a stable alias → component map so that the loader
    can apply text encoder overrides.
    """
    register_text_encoder(context, "qwen3_4b", "text_encoder")


def build_plan(signature: ModelSignature) -> ParserPlanBundle:
    # GGUF core-only checkpoints: only the DiT backbone lives in the
    # state_dict, while Qwen3/VAE come from external paths.
    if signature.quantization.kind == QuantizationKind.GGUF:
        plan = ParserPlan(
            splits=[
                # Core-only: include all tensors under the single transformer component.
                SplitSpec(name="transformer", prefixes=("",)),
            ],
            converters=(),
            validations=(
                ValidationSpec(name="register_zimage_text_encoders", function=_register_zimage_text_encoders),
                ValidationSpec(name="core_presence", function=_validate_transformer_core),
                ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
            ),
        )
        return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))

    # Full/bf16 checkpoints: expect transformer + VAE + text encoder.
    plan = ParserPlan(
        splits=[
            SplitSpec(name="transformer", prefixes=_ZIMAGE_CORE_PREFIXES),
            SplitSpec(name="vae", prefixes=("vae.",), required=False),
            SplitSpec(name="text_encoder", prefixes=("text_encoder.",), required=False),
        ],
        converters=(),
        validations=(
            ValidationSpec(name="register_zimage_text_encoders", function=_register_zimage_text_encoders),
            ValidationSpec(name="core_presence", function=_validate_transformer_core),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))


def _validate_transformer_core(context):
    """Validate that the transformer core has key blocks."""
    unet = context.require("transformer").tensors
    # Check for NextDiT/Lumina2 style keys with various prefixes
    base_key = "layers.0.adaLN_modulation.0.weight"
    possible_keys = [
        base_key,  # GGUF format
        f"model.{base_key}",  # Some safetensors
        f"model.diffusion_model.{base_key}",  # FP8/BF16 safetensors
    ]
    if not any(k in unet for k in possible_keys):
        raise ValidationError("Z Image transformer missing adaLN modulation layer", component="transformer")
