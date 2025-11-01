from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any, List, Mapping, Optional

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.sd.spec import SD35_SPEC, SDEngineRuntime, assemble_engine_runtime
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models.loader import DiffusionModelBundle


logger = logging.getLogger("backend.engines.sd.sd35")


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
        runtime = assemble_engine_runtime(SD35_SPEC, bundle.estimated_config, bundle.components)
        self._runtime = runtime
        self.register_model_family("sd3")

        logger.debug("StableDiffusion3 runtime prepared with classic branches=%s", runtime.classic_order)

        return CodexObjects(
            unet=runtime.unet,
            clip=runtime.clip,
            vae=runtime.vae,
            clipvision=None,
        )

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
        memory_management.load_model_gpu(self.codex_objects.clip.patcher)

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

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        engine = runtime.t5_engine("t5xxl")
        token_count = len(engine.tokenize([prompt])[0])
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
