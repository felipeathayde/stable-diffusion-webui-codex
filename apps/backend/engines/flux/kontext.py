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

import logging
from typing import Any, Mapping

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexObjects
from apps.backend.engines.flux.flux import Flux
from apps.backend.engines.flux.factory import CodexFluxFamilyFactory
from apps.backend.engines.flux.spec import FLUX_SPEC
from apps.backend.runtime.models.loader import DiffusionModelBundle

logger = logging.getLogger("backend.engines.flux.kontext")

_KONTEXT_FACTORY = CodexFluxFamilyFactory(spec=FLUX_SPEC)


class Kontext(Flux):
    """Flux Kontext engine (Flux-derived, image-conditioned)."""

    engine_id = "flux1_kontext"

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

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        assembly = _KONTEXT_FACTORY.assemble(bundle, options=options)
        runtime = assembly.runtime
        self._runtime = runtime
        self.use_distilled_cfg_scale = runtime.use_distilled_cfg
        logger.debug("Kontext runtime prepared (distilled cfg=%s)", runtime.use_distilled_cfg)

        from apps.backend.runtime.families.flux.streaming import StreamedFluxCore

        core_model = getattr(runtime.denoiser.model, "diffusion_model", runtime.denoiser.model)
        if isinstance(core_model, StreamedFluxCore):
            self._streaming_controller = core_model.controller
        else:
            self._streaming_controller = None

        return assembly.codex_objects
