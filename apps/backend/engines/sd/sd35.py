"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Codex-native Stable Diffusion 3.5 engine.
Assembles an `SDEngineRuntime` using `SD35_SPEC`; optional text-encoder behavior is controlled via `CODEX_SD3_ENABLE_T5`.

Symbols (top-level; keep in sync; no ghosts):
- `_opts` (function): Loads SD3/SD35 environment flags (currently `CODEX_SD3_ENABLE_T5`) into a simple namespace.
- `StableDiffusion3` (class): SD 3.5 diffusion engine wiring runtime components to the Codex engine interface.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any, List, Mapping, Optional

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.sd.factory import CodexSDFamilyFactory
from apps.backend.engines.sd.spec import SD35_SPEC, SDEngineRuntime
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models.loader import DiffusionModelBundle


logger = logging.getLogger("backend.engines.sd.sd35")

_SD35_FACTORY = CodexSDFamilyFactory(spec=SD35_SPEC)


def _opts():
    enable_t5 = os.getenv("CODEX_SD3_ENABLE_T5", "1").lower() in ("1", "true", "yes", "on")
    return SimpleNamespace(sd3_enable_t5=enable_t5)


class StableDiffusion3(CodexDiffusionEngine):
    """Codex-native Stable Diffusion 3 engine."""

    engine_id = "sd35"

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[SDEngineRuntime] = None

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG, TaskType.IMG2IMG),
            model_types=("sd35", "sd3"),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
        )

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        assembly = _SD35_FACTORY.assemble(bundle, options=dict(options))
        runtime = assembly.runtime
        self._runtime = runtime
        self.register_model_family("sd3")

        logger.debug("StableDiffusion3 runtime prepared with classic branches=%s", runtime.classic_order)

        return assembly.codex_objects

    def _on_unload(self) -> None:
        self._runtime = None

    def _require_runtime(self) -> SDEngineRuntime:
        if self._runtime is None:
            raise RuntimeError("StableDiffusion3 runtime is not initialised; call load() first.")
        return self._runtime

    def set_clip_skip(self, clip_skip: int):
        runtime = self._require_runtime()
        runtime.set_clip_skip(clip_skip)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: List[str]):
        runtime = self._require_runtime()
        memory_management.load_model_gpu(self.codex_objects.text_encoders["clip"].patcher)
        unload_clip = self.smart_offload_enabled
        try:
            cond_l, pooled_l = runtime.classic_engine("clip_l")(prompt)
            cond_g, pooled_g = runtime.classic_engine("clip_g")(prompt)

            opts = _opts()
            if opts.sd3_enable_t5:
                cond_t5 = runtime.t5_engine("t5xxl")(prompt)
            else:
                cond_t5 = torch.zeros((len(prompt), 256, 4096), device=cond_l.device, dtype=cond_l.dtype)

            is_negative_prompt = getattr(prompt, "is_negative_prompt", False)
            force_zero_negative_prompt = is_negative_prompt and all(x == "" for x in prompt)

            if force_zero_negative_prompt:
                pooled_l = torch.zeros_like(pooled_l)
                pooled_g = torch.zeros_like(pooled_g)
                cond_l = torch.zeros_like(cond_l)
                cond_g = torch.zeros_like(cond_g)
                cond_t5 = torch.zeros_like(cond_t5)

            cond_lg = torch.cat([cond_l, cond_g], dim=-1)
            if cond_lg.shape[-1] < 4096:
                pad = 4096 - cond_lg.shape[-1]
                cond_lg = torch.nn.functional.pad(cond_lg, (0, pad))

            cond = {
                "crossattn": torch.cat([cond_lg, cond_t5], dim=-2),
                "vector": torch.cat([pooled_l, pooled_g], dim=-1),
            }

            return cond
        finally:
            if unload_clip:
                memory_management.unload_model(self.codex_objects.text_encoders["clip"].patcher)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        engine = runtime.t5_engine("t5xxl")
        token_count = len(engine.tokenize([prompt])[0])
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
