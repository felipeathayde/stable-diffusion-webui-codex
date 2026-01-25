"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Codex-native Stable Diffusion 2.x engine.
Assembles an `SDEngineRuntime` using `SD20_SPEC` and exposes the `CodexDiffusionEngine` surface (txt2img/img2img).

Symbols (top-level; keep in sync; no ghosts):
- `StableDiffusion2` (class): SD 2.x diffusion engine wiring runtime components to the Codex engine interface.
"""

from __future__ import annotations


from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.sd.factory import CodexSDFamilyFactory
from apps.backend.engines.sd.classic_base import CodexSDClassicEngineBase
from apps.backend.engines.sd.spec import SD20_SPEC
from apps.backend.runtime.model_registry.specs import ModelFamily

_SD20_FACTORY = CodexSDFamilyFactory(spec=SD20_SPEC)


class StableDiffusion2(CodexSDClassicEngineBase):
    """Codex-native Stable Diffusion 2.x engine."""

    engine_id = "sd20"
    expected_family = ModelFamily.SD20
    _factory = _SD20_FACTORY
    _model_family = "sd2"

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG, TaskType.IMG2IMG),
            model_types=("sd20",),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
        )
