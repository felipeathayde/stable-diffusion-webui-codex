"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared SD1/SD2 engine implementation (classic CLIP text encoder + UNet denoiser + VAE).
Centralizes the duplicated conditioning/encode/decode logic so SD15 and SD20 engines only specify spec + capabilities.

Symbols (top-level; keep in sync; no ghosts):
- `CodexSDClassicEngineBase` (class): Shared implementation for SD1/SD2 engines (build/runtime lifecycle + TE/VAE helpers).
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional

import torch

from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.sd.factory import CodexSDFamilyFactory
from apps.backend.engines.sd.spec import SDEngineRuntime
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models.loader import DiffusionModelBundle


class CodexSDClassicEngineBase(CodexDiffusionEngine):
    """Shared SD1/SD2 engine implementation (classic CLIP + UNet + VAE)."""

    engine_id: str
    _factory: CodexSDFamilyFactory
    _model_family: str

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[SDEngineRuntime] = None
        self._primary_branch: Optional[str] = None

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(f"backend.engines.sd.{self.engine_id}")

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        assembly = self._factory.assemble(bundle, options=dict(options))
        runtime = assembly.runtime
        self._runtime = runtime
        self._primary_branch = runtime.classic_order[0] if runtime.classic_order else None
        self.register_model_family(self._model_family)

        self._logger.debug("%s runtime prepared with branches=%s", self.engine_id, runtime.classic_order)
        return assembly.codex_objects

    def _on_unload(self) -> None:
        self._runtime = None
        self._primary_branch = None

    def _require_runtime(self) -> SDEngineRuntime:
        if self._runtime is None:
            raise RuntimeError(f"{self.engine_id} runtime is not initialised; call load() first.")
        return self._runtime

    def set_clip_skip(self, clip_skip: int) -> None:
        runtime = self._require_runtime()
        runtime.set_clip_skip(clip_skip)
        self._logger.debug("Clip skip set to %d for %s.", clip_skip, self.engine_id)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: List[str]):
        runtime = self._require_runtime()
        memory_management.manager.load_model(self.codex_objects.text_encoders["clip"].patcher)
        unload_clip = self.smart_offload_enabled
        try:
            conditioning = runtime.primary_classic()(prompt)
            self._logger.debug("Generated conditioning for %d prompts.", len(prompt))
            return conditioning
        finally:
            if unload_clip:
                memory_management.manager.unload_model(self.codex_objects.text_encoders["clip"].patcher)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        engine = runtime.primary_classic()
        _, token_count = engine.process_texts([prompt])
        target = engine.get_target_prompt_token_count(token_count)
        return token_count, target

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        memory_management.manager.load_model(self.codex_objects.vae)
        unload_vae = self.smart_offload_enabled
        try:
            sample = self.codex_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
            sample = self.codex_objects.vae.first_stage_model.process_in(sample)
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.manager.unload_model(self.codex_objects.vae)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        memory_management.manager.load_model(self.codex_objects.vae)
        unload_vae = self.smart_offload_enabled
        try:
            sample = self.codex_objects.vae.first_stage_model.process_out(x)
            sample = self.codex_objects.vae.decode(sample)
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.manager.unload_model(self.codex_objects.vae)
