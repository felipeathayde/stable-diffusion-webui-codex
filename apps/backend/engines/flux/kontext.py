"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux.1 Kontext engine implementation (Flux-derived, image-conditioned flow model).
Assembles the Kontext runtime via `CodexFluxFamilyFactory` and exposes engine capabilities. Kontext img2img semantics
(init image as `image_latents`) are handled by the canonical img2img use-case (`apps/backend/use_cases/img2img.py`).

Symbols (top-level; keep in sync; no ghosts):
- `Kontext` (class): Flux-derived image-conditioned engine.
"""

from __future__ import annotations

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.flux.flux import Flux
from apps.backend.runtime.model_registry.specs import ModelFamily


class Kontext(Flux):
    """Flux Kontext engine (Flux-derived, image-conditioned)."""

    engine_id = "flux1_kontext"
    expected_family = ModelFamily.FLUX_KONTEXT

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG, TaskType.IMG2IMG),
            model_types=("flux1_kontext", "kontext"),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
            extras={
                "samplers": ("euler", "euler a", "ddim", "dpm++ 2m"),
                "schedulers": ("simple", "beta", "normal"),
            },
        )
