from __future__ import annotations

import logging
from typing import Iterable, List

import torch

from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.flux.spec import FLUX_SPEC, assemble_flux_runtime
from apps.backend.runtime.memory import memory_management

logger = logging.getLogger("backend.engines.flux")


class Flux(CodexDiffusionEngine):
    """Codex native Flux engine."""

    def __init__(self, estimated_config, codex_components):
        super().__init__(estimated_config, codex_components)
        runtime = assemble_flux_runtime(spec=FLUX_SPEC, estimated_config=estimated_config, codex_components=codex_components)
        self._runtime = runtime
        self._guidance_default = FLUX_SPEC.distilled_cfg_scale_default

        self.bind_components(
            CodexObjects(
                unet=runtime.unet,
                clip=runtime.clip,
                vae=runtime.vae,
                clipvision=None,
            ),
            label="flux",
        )
        self.use_distilled_cfg_scale = runtime.use_distilled_cfg
        logger.debug("Flux engine initialised (distilled cfg=%s)", self.use_distilled_cfg_scale)

    def set_clip_skip(self, clip_skip: int) -> None:
        self._runtime.set_clip_skip(clip_skip)
        logger.debug("Flux clip skip set to %d", clip_skip)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: List[str]):
        memory_management.load_model_gpu(self.codex_objects.clip.patcher)
        clip_branch = self._runtime.text.clip_text
        cond_l, pooled_l = (clip_branch(prompt) if clip_branch is not None else (None, None))
        cond_t5 = self._runtime.text.t5_text(prompt)
        cond = {"crossattn": cond_t5}

        if pooled_l is not None:
            cond["vector"] = pooled_l

        if self.use_distilled_cfg_scale:
            distilled_cfg_scale = getattr(prompt, "distilled_cfg_scale", self._guidance_default) or self._guidance_default
            cond["guidance"] = torch.full((len(prompt),), float(distilled_cfg_scale), dtype=torch.float32)
            logger.debug("Flux distilled cfg scale=%s", distilled_cfg_scale)
        else:
            logger.debug("Flux distilled cfg disabled (schnell variant)")

        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        token_count = len(self._runtime.text.t5_text.tokenize([prompt])[0])
        return token_count, max(255, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        sample = self.codex_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.codex_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        sample = self.codex_objects.vae.first_stage_model.process_out(x)
        sample = self.codex_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)
