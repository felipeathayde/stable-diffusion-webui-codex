"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Stable Diffusion 3 / 3.5 model detector for the model registry.
Detects SD3-family checkpoints and builds a `ModelSignature`, heuristically classifying SD3 vs SD3.5 (medium/large) from transformer depth/structure.

Symbols (top-level; keep in sync; no ghosts):
- `SD3_REQUIRED_KEYS` (constant): Key set used to identify SD3-family checkpoints.
- `StableDiffusion3Detector` (class): Detector for SD3/SD3.5 checkpoints (variant classification via `_classify_variant`).
- `_shape` (function): Helper to read a single shape dimension from a bundle.
- `_classify_variant` (function): Infers SD3/SD3.5 variant from depth and dual-block presence.
"""

from __future__ import annotations

from apps.backend.runtime.model_registry.detectors.base import REGISTRY
from apps.backend.runtime.model_registry.signals import SignalBundle, count_blocks, has_all_keys
from apps.backend.runtime.model_registry.specs import (
    CodexCoreArchitecture,
    CodexCoreSignature,
    LatentFormat,
    ModelFamily,
    ModelSignature,
    PredictionKind,
    QuantizationHint,
    TextEncoderSignature,
    VAESignature,
)


SD3_REQUIRED_KEYS = (
    "joint_blocks.0.context_block.attn.qkv.weight",
    "x_embedder.proj.weight",
    "final_layer.linear.weight",
)


class StableDiffusion3Detector:
    priority = 160

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        return has_all_keys(bundle, *SD3_REQUIRED_KEYS)

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        channels_in = _shape(bundle, "x_embedder.proj.weight", 1) or 16
        channels_out = _shape(bundle, "final_layer.linear.weight", 0) or channels_in
        context_dim = _shape(bundle, "context_embedder.weight", 0) or 4096
        depth = count_blocks(bundle.keys, "joint_blocks.{}.")
        has_dual_blocks = bundle.has_prefix("dual_blocks.")

        variant, family, repo = _classify_variant(depth=depth, has_dual_blocks=has_dual_blocks)

        text_encoders = [
            TextEncoderSignature(
                name="clip_l",
                key_prefix="text_encoders.clip_l.",
                expected_dim=768,
            ),
            TextEncoderSignature(
                name="clip_g",
                key_prefix="text_encoders.clip_g.",
                expected_dim=1280,
            ),
            TextEncoderSignature(
                name="t5xxl",
                key_prefix="text_encoders.t5xxl.",
                expected_dim=4096,
            ),
        ]

        vae = None
        if any(key.startswith("vae.") for key in bundle.keys):
            vae = VAESignature(key_prefix="vae.", latent_channels=16)

        extras = {
            "joint_blocks": depth,
            "variant": variant,
        }

        return ModelSignature(
            family=family,
            repo_hint=repo,
            prediction=PredictionKind.EPSILON,
            latent_format=LatentFormat.SD_3,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.DIT,
                channels_in=channels_in,
                channels_out=channels_out,
                context_dim=context_dim,
                temporal=False,
                depth=depth,
                key_prefixes=["joint_blocks."],
            ),
            text_encoders=text_encoders,
            vae=vae,
            extras=extras,
        )


def _shape(bundle: SignalBundle, key: str, dim: int) -> int | None:
    shape = bundle.shape(key)
    if not shape or len(shape) <= dim:
        return None
    return int(shape[dim])


def _classify_variant(*, depth: int, has_dual_blocks: bool) -> tuple[str, ModelFamily, str]:
    """Infer SD3/SD3.5 variant heuristically from transformer depth & structure."""
    if depth >= 35:
        return "sd35_large", ModelFamily.SD35, "stabilityai/stable-diffusion-3.5-large"
    if depth >= 24 and has_dual_blocks:
        return "sd35_medium", ModelFamily.SD35, "stabilityai/stable-diffusion-3.5-medium"
    return "sd3_medium", ModelFamily.SD3, "stabilityai/stable-diffusion-3-medium-diffusers"


REGISTRY.register(StableDiffusion3Detector())
