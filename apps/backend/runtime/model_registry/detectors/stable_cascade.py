"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Stable Cascade stage B/C model detectors for the model registry.
Detects Stable Cascade prior/decoder checkpoints by locating stage-specific mapper keys, then builds `ModelSignature` metadata for loader/UI inventory.

Symbols (top-level; keep in sync; no ghosts):
- `StableCascadeStageCDetector` (class): Detector for Stable Cascade stage C checkpoints (prior).
- `StableCascadeStageBDetector` (class): Detector for Stable Cascade stage B checkpoints (decoder; lite/full variant heuristic).
- `_prefix_from_key` (function): Derives the shared stage prefix from a discovered key.
- `_find_key` (function): Searches for a key suffix across common candidate prefixes.
"""

from __future__ import annotations

from typing import Optional

from apps.backend.runtime.model_registry.detectors.base import ModelDetector, REGISTRY
from apps.backend.runtime.model_registry.signals import SignalBundle, count_blocks
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


class StableCascadeStageCDetector:
    priority = 190

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        return _find_key(bundle, "clip_txt_mapper.weight") is not None

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        txt_key = _find_key(bundle, "clip_txt_mapper.weight")
        prefix = _prefix_from_key(txt_key)
        cond_key = f"{prefix}clip_txt_mapper.weight"
        cond_shape = bundle.shape(cond_key)
        cond_dim = cond_shape[0] if cond_shape else 2048

        down_blocks = count_blocks(bundle.keys, f"{prefix}down_blocks.{{}}.")
        extras = {"stage": "C", "clip_txt_dim": cond_dim}

        return ModelSignature(
            family=ModelFamily.STABLE_CASCADE,
            repo_hint="stabilityai/stable-cascade-prior",
            prediction=PredictionKind.EPSILON,
            latent_format=LatentFormat.CASCADE,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.DIT,
                channels_in=16,
                channels_out=16,
                context_dim=2048,
                temporal=False,
                depth=down_blocks,
                key_prefixes=[prefix or ""],
            ),
            text_encoders=[
                TextEncoderSignature(
                    name="clip_g",
                    key_prefix="text_encoder.clip_g.",
                    expected_dim=1280,
                )
            ],
            vae=None,
            extras=extras,
        )


class StableCascadeStageBDetector:
    priority = 195

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        key = _find_key(bundle, "clip_mapper.weight")
        if key is None:
            return False
        # Ensure stage C is not double-counted
        return _find_key(bundle, "clip_txt_mapper.weight") is None

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        mapper_key = _find_key(bundle, "clip_mapper.weight")
        prefix = _prefix_from_key(mapper_key)
        repeat_tensor = bundle.shape(f"{prefix}down_blocks.1.0.channelwise.0.weight")
        variant = "full"
        if repeat_tensor:
            hidden = repeat_tensor[-1]
            if hidden == 576:
                variant = "lite"
        down_blocks = count_blocks(bundle.keys, f"{prefix}down_blocks.{{}}.")
        extras = {"stage": "B", "variant": variant}

        vae = None
        if _find_key(bundle, "vae.decoder.conv_out.weight"):
            vae = VAESignature(key_prefix="vae.", latent_channels=4)

        return ModelSignature(
            family=ModelFamily.STABLE_CASCADE,
            repo_hint="stabilityai/stable-cascade",
            prediction=PredictionKind.EPSILON,
            latent_format=LatentFormat.CASCADE,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.DIT,
                channels_in=4,
                channels_out=4,
                context_dim=1280,
                temporal=False,
                depth=down_blocks,
                key_prefixes=[prefix or ""],
            ),
            text_encoders=[
                TextEncoderSignature(
                    name="clip_g",
                    key_prefix="text_encoder.clip_g.",
                    expected_dim=1280,
                )
            ],
            vae=vae,
            extras=extras,
        )


def _prefix_from_key(key: Optional[str]) -> str:
    if not key:
        return ""
    return key[: key.rfind("clip")]


def _find_key(bundle: SignalBundle, suffix: str) -> Optional[str]:
    candidates = (
        suffix,
        f"model.{suffix}",
        f"model.diffusion_model.{suffix}",
        f"prior.{suffix}",
        f"decoder.{suffix}",
    )
    for key in candidates:
        if key in bundle.state_dict:
            return key
    return None


REGISTRY.register(StableCascadeStageCDetector())
REGISTRY.register(StableCascadeStageBDetector())
