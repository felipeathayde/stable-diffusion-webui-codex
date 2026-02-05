"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima parser plan builder (Cosmos Predict2 / MiniTrainDiT core-only checkpoints).
Builds split and validation steps for Anima core checkpoints and registers stable text encoder alias mappings so the
loader can apply sha-selected overrides deterministically.

Symbols (top-level; keep in sync; no ghosts):
- `_register_anima_text_encoders` (function): Registers the `qwen3_06b` alias mapping in the parser context.
- `_validate_anima_transformer_core` (function): Validates presence of key Anima core tensors after prefix stripping.
- `build_plan` (function): Builds and returns the Anima `ParserPlanBundle`.
"""

from __future__ import annotations

from apps.backend.runtime.model_registry.specs import ModelSignature

from ..builders import build_estimated_config, register_text_encoder
from ..errors import ValidationError
from ..quantization import validate_component_dtypes
from ..specs import ParserPlan, ParserPlanBundle, SplitSpec, ValidationSpec


def _register_anima_text_encoders(context) -> None:  # type: ignore[no-untyped-def]
    """Register expected Anima text encoder aliases even when weights are external.

    Anima core checkpoints are exported as `net.*` (Cosmos Predict2 / MiniTrainDiT) without
    embedding the Qwen3 text encoder weights. We still need a stable alias → component
    mapping so that `tenc_path` shorthand and TE override resolution can validate aliases.
    """

    register_text_encoder(context, "qwen3_06b", "text_encoder")


def _validate_anima_transformer_core(context) -> None:  # type: ignore[no-untyped-def]
    core = context.require("transformer").tensors
    required = (
        "x_embedder.proj.1.weight",
        "t_embedder.1.linear_1.weight",
        "blocks.0.self_attn.q_proj.weight",
        "blocks.0.cross_attn.k_proj.weight",
        "final_layer.linear.weight",
    )
    missing = [k for k in required if k not in core]
    if missing:
        raise ValidationError(
            "Anima core transformer is missing required tensors; sample=%s" % ", ".join(missing[:5]),
            component="transformer",
        )


def build_plan(signature: ModelSignature) -> ParserPlanBundle:
    plan = ParserPlan(
        splits=[
            # Core-only: include all tensors under the single transformer component; strip the `net.` prefix.
            SplitSpec(name="transformer", prefixes=("net.",)),
        ],
        converters=(),
        validations=(
            ValidationSpec(name="register_anima_text_encoders", function=_register_anima_text_encoders),
            ValidationSpec(name="core_presence", function=_validate_anima_transformer_core),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: build_estimated_config(ctx, signature))

