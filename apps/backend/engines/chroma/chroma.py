"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Codex-native Chroma engine built on the Flux engine toolkit.
Assembles a `FluxEngineRuntime` from `CHROMA_SPEC` via the Flux family factory and exposes the `CodexDiffusionEngine` surface used by API/use-cases.

Symbols (top-level; keep in sync; no ghosts):
- `Chroma` (class): Chroma diffusion engine (txt2img/img2img) wiring the Chroma runtime (Flux toolkit) to the Codex engine interface.
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.flux.factory import CodexFluxFamilyFactory
from apps.backend.engines.flux.spec import CHROMA_SPEC, FluxEngineRuntime
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models.loader import DiffusionModelBundle

logger = logging.getLogger("backend.engines.chroma")

_CHROMA_FACTORY = CodexFluxFamilyFactory(spec=CHROMA_SPEC)


class Chroma(CodexDiffusionEngine):
    """Codex native Chroma engine built on Flux toolkit."""

    engine_id = "chroma"

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[FluxEngineRuntime] = None

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG, TaskType.IMG2IMG),
            model_types=("chroma",),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
        )

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        assembly = _CHROMA_FACTORY.assemble(bundle, options=options)
        runtime = assembly.runtime
        self._runtime = runtime
        self.use_distilled_cfg_scale = runtime.use_distilled_cfg
        logger.debug("Chroma runtime prepared")

        return assembly.codex_objects

    def _on_unload(self) -> None:
        self._runtime = None

    def _require_runtime(self) -> FluxEngineRuntime:
        if self._runtime is None:
            raise RuntimeError("Chroma runtime is not initialised; call load() first.")
        return self._runtime

    def set_clip_skip(self, clip_skip: int):
        logger.debug("Chroma ignores clip_skip (no CLIP branch)")

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: List[str]):
        runtime = self._require_runtime()
        memory_management.load_model_gpu(self.codex_objects.text_encoders["clip"].patcher)
        unload_clip = self.smart_offload_enabled
        try:
            conditioning = runtime.text.t5_text(prompt)
            logger.debug("Chroma conditioning generated for %d prompts", len(prompt))
            return conditioning
        finally:
            if unload_clip:
                memory_management.unload_model(self.codex_objects.text_encoders["clip"].patcher)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        token_count = len(runtime.text.t5_text.tokenize([prompt])[0])
        return token_count, max(255, token_count)

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
