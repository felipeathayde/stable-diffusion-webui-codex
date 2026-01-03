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

import logging
from typing import Any, List, Mapping, Optional

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.sd.spec import SD15_SPEC, SDEngineRuntime, assemble_engine_runtime
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models.loader import DiffusionModelBundle

logger = logging.getLogger("backend.engines.sd.sd15")


class StableDiffusion(CodexDiffusionEngine):
    """Codex-native Stable Diffusion 1.x engine."""

    engine_id = "sd15"

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[SDEngineRuntime] = None
        self._primary_branch: Optional[str] = None

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG, TaskType.IMG2IMG),
            model_types=("sd15",),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
        )

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        runtime = assemble_engine_runtime(SD15_SPEC, bundle.estimated_config, bundle.components)
        self._runtime = runtime
        self._primary_branch = runtime.classic_order[0] if runtime.classic_order else None
        self.register_model_family("sd1")

        logger.debug(
            "StableDiffusion runtime prepared with branches=%s",
            runtime.classic_order,
        )

        return CodexObjects(
            unet=runtime.unet,
            vae=runtime.vae,
            text_encoders={"clip": runtime.clip},
            clipvision=None,
        )

    def _on_unload(self) -> None:
        self._runtime = None
        self._primary_branch = None

    def _require_runtime(self) -> SDEngineRuntime:
        if self._runtime is None:
            raise RuntimeError("StableDiffusion runtime is not initialised; call load() first.")
        return self._runtime

    def set_clip_skip(self, clip_skip: int) -> None:
        runtime = self._require_runtime()
        runtime.set_clip_skip(clip_skip)
        logger.debug("Clip skip set to %d for SD15.", clip_skip)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: List[str]):
        runtime = self._require_runtime()
        memory_management.load_model_gpu(self.codex_objects.text_encoders["clip"].patcher)
        unload_clip = self.smart_offload_enabled
        try:
            conditioning = runtime.primary_classic()(prompt)
            logger.debug("Generated conditioning for %d prompts (SD15).", len(prompt))
            return conditioning
        finally:
            if unload_clip:
                memory_management.unload_model(self.codex_objects.text_encoders["clip"].patcher)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        engine = runtime.primary_classic()
        _, token_count = engine.process_texts([prompt])
        target = engine.get_target_prompt_token_count(token_count)
        return token_count, target

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        memory_management.load_model_gpu(self.codex_objects.vae)
        unload_vae = self.smart_offload_enabled
        try:
            sample = self.codex_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
            sample = self.codex_objects.vae.first_stage_model.process_in(sample)
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.unload_model(self.codex_objects.vae)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        memory_management.load_model_gpu(self.codex_objects.vae)
        unload_vae = self.smart_offload_enabled
        try:
            sample = self.codex_objects.vae.first_stage_model.process_out(x)
            sample = self.codex_objects.vae.decode(sample)
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.unload_model(self.codex_objects.vae)
