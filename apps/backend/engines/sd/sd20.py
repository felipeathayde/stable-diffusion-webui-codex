from __future__ import annotations

import logging
from typing import List

import torch

from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.sd.spec import SD20_SPEC, assemble_engine_runtime
from apps.backend.runtime.memory import memory_management

logger = logging.getLogger("backend.engines.sd.sd20")


class StableDiffusion2(CodexDiffusionEngine):
    """Codex-native Stable Diffusion 2.x engine."""

    def __init__(self, estimated_config, codex_components):
        super().__init__(estimated_config, codex_components)

        runtime = assemble_engine_runtime(SD20_SPEC, estimated_config, codex_components)
        self._runtime = runtime
        self._primary_branch = runtime.classic_order[0]

        self.bind_components(
            CodexObjects(
                unet=runtime.unet,
                clip=runtime.clip,
                vae=runtime.vae,
                clipvision=None,
            ),
            label="sd20",
        )
        self.register_model_family("sd2")

        logger.debug(
            "StableDiffusion2 initialised with branches=%s clip_skip=%d",
            runtime.classic_order,
            runtime.classic_engine(self._primary_branch).clip_skip,
        )

    def set_clip_skip(self, clip_skip: int) -> None:
        self._runtime.set_clip_skip(clip_skip)
        logger.debug("Clip skip set to %d for all branches.", clip_skip)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: List[str]):
        memory_management.load_model_gpu(self.codex_objects.clip.patcher)
        conditioning = self._runtime.primary_classic()(prompt)
        logger.debug("Generated conditioning for %d prompts.", len(prompt))
        return conditioning

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        engine = self._runtime.primary_classic()
        _, token_count = engine.process_texts([prompt])
        target = engine.get_target_prompt_token_count(token_count)
        return token_count, target

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
