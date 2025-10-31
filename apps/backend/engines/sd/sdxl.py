from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Iterable, List, Tuple

import torch

from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.sd.spec import SDXL_REFINER_SPEC, SDXL_SPEC, assemble_engine_runtime
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.common.nn.unet import Timestep

logger = logging.getLogger("backend.engines.sd.sdxl")


def _opts() -> SimpleNamespace:
    return SimpleNamespace(
        sdxl_crop_left=0,
        sdxl_crop_top=0,
        sdxl_refiner_low_aesthetic_score=2.5,
        sdxl_refiner_high_aesthetic_score=6.0,
    )


def _prompt_meta(prompt: Iterable[str]) -> Tuple[int, int, bool]:
    obj = prompt  # type: ignore
    width = getattr(obj, "width", 1024) or 1024
    height = getattr(obj, "height", 1024) or 1024
    is_negative = getattr(obj, "is_negative_prompt", False)
    return width, height, is_negative


class StableDiffusionXL(CodexDiffusionEngine):
    """Codex-native SDXL base engine."""

    def __init__(self, estimated_config, codex_components):
        super().__init__(estimated_config, codex_components)

        runtime = assemble_engine_runtime(SDXL_SPEC, estimated_config, codex_components)
        self._runtime = runtime

        self.embedder = Timestep(256)

        self.codex_objects = CodexObjects(
            unet=runtime.unet,
            clip=runtime.clip,
            vae=runtime.vae,
            clipvision=None,
        )
        self.codex_objects_original = self.codex_objects.shallow_copy()
        self.codex_objects_after_applying_lora = self.codex_objects.shallow_copy()

        self.is_sdxl = True

        logger.debug(
            "StableDiffusionXL initialised with branches=%s clip_skip=%d",
            runtime.branch_order,
            runtime.text_engine("clip_l").clip_skip,
        )

    def set_clip_skip(self, clip_skip: int) -> None:
        self._runtime.set_clip_skip(clip_skip)
        logger.debug("Clip skip set to %d for SDXL.", clip_skip)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: List[str]):
        memory_management.load_model_gpu(self.codex_objects.clip.patcher)

        cond_l = self._runtime.text_engine("clip_l")(prompt)
        cond_g, pooled = self._runtime.text_engine("clip_g")(prompt)

        width, height, is_negative = _prompt_meta(prompt)
        opts = _opts()

        embed_values = [
            self.embedder(torch.tensor([height])),
            self.embedder(torch.tensor([width])),
            self.embedder(torch.tensor([opts.sdxl_crop_top])),
            self.embedder(torch.tensor([opts.sdxl_crop_left])),
            self.embedder(torch.tensor([height])),
            self.embedder(torch.tensor([width])),
        ]

        flat = torch.flatten(torch.cat(embed_values)).unsqueeze(dim=0).repeat(pooled.shape[0], 1).to(pooled)

        if is_negative and all(x == "" for x in prompt):
            pooled = torch.zeros_like(pooled)
            cond_l = torch.zeros_like(cond_l)
            cond_g = torch.zeros_like(cond_g)

        cond = {
            "crossattn": torch.cat([cond_l, cond_g], dim=2),
            "vector": torch.cat([pooled, flat], dim=1),
        }

        logger.debug("Generated SDXL conditioning for %d prompts.", len(prompt))
        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        engine = self._runtime.text_engine("clip_l")
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



class StableDiffusionXLRefiner(CodexDiffusionEngine):
    """Codex-native SDXL refiner engine."""

    def __init__(self, estimated_config, codex_components):
        super().__init__(estimated_config, codex_components)

        runtime = assemble_engine_runtime(SDXL_REFINER_SPEC, estimated_config, codex_components)
        self._runtime = runtime

        self.embedder = Timestep(256)

        self.codex_objects = CodexObjects(
            unet=runtime.unet,
            clip=runtime.clip,
            vae=runtime.vae,
            clipvision=None,
        )
        self.codex_objects_original = self.codex_objects.shallow_copy()
        self.codex_objects_after_applying_lora = self.codex_objects.shallow_copy()

        self.is_sdxl = True

        logger.debug("StableDiffusionXLRefiner initialised with clip_skip=%d", runtime.text_engine("clip_g").clip_skip)

    def set_clip_skip(self, clip_skip: int) -> None:
        self._runtime.set_clip_skip(clip_skip)
        logger.debug("Clip skip set to %d for SDXL refiner.", clip_skip)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: List[str]):
        memory_management.load_model_gpu(self.codex_objects.clip.patcher)

        cond_g, pooled = self._runtime.text_engine("clip_g")(prompt)

        width, height, is_negative = _prompt_meta(prompt)
        opts = _opts()

        embed_values = [
            self.embedder(torch.tensor([height])),
            self.embedder(torch.tensor([width])),
            self.embedder(torch.tensor([opts.sdxl_crop_top])),
            self.embedder(torch.tensor([opts.sdxl_crop_left])),
            self.embedder(torch.tensor([height])),
            self.embedder(torch.tensor([width])),
        ]
        flat = torch.flatten(torch.cat(embed_values)).unsqueeze(dim=0).repeat(pooled.shape[0], 1).to(pooled)

        if is_negative and all(x == "" for x in prompt):
            pooled = torch.zeros_like(pooled)
            cond_g = torch.zeros_like(cond_g)

        cond = {
            "crossattn": cond_g,
            "vector": torch.cat([pooled, flat], dim=1),
        }

        logger.debug("Generated SDXL refiner conditioning for %d prompts.", len(prompt))
        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        engine = self._runtime.text_engine("clip_g")
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
