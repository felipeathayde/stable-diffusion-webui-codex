"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Codex-native Stable Diffusion 1.5 engine.
Assembles an `SDEngineRuntime` using `SD15_SPEC` and exposes the `CodexDiffusionEngine` surface (txt2img/img2img).

Symbols (top-level; keep in sync; no ghosts):
- `StableDiffusion` (class): SD 1.5 diffusion engine wiring runtime components to the Codex engine interface.
"""

from __future__ import annotations


from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.sd.factory import CodexSDFamilyFactory
from apps.backend.engines.sd.classic_base import CodexSDClassicEngineBase
from apps.backend.engines.sd.spec import SD15_SPEC
from apps.backend.runtime.model_registry.specs import ModelFamily

_SD15_FACTORY = CodexSDFamilyFactory(spec=SD15_SPEC)


class StableDiffusion(CodexSDClassicEngineBase):
    """Codex-native Stable Diffusion 1.x engine."""

    engine_id = "sd15"
    expected_family = ModelFamily.SD15
    _factory = _SD15_FACTORY
    _model_family = "sd1"

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG, TaskType.IMG2IMG),
            model_types=("sd15",),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
        )
