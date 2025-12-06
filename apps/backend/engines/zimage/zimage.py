"""Z Image Turbo Engine.

Alibaba Z Image Turbo (6B) txt2img engine following Flux pattern.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Mapping, Optional

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent, Txt2ImgRequest
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models.loader import DiffusionModelBundle

from .spec import ZIMAGE_SPEC, ZImageEngineRuntime, assemble_zimage_runtime

logger = logging.getLogger("backend.engines.zimage.zimage")


class ZImageEngine(CodexDiffusionEngine):
    """Z Image Turbo txt2img engine."""

    engine_id = "zimage"

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[ZImageEngineRuntime] = None
        self._device = "cuda"
        self._dtype = "bf16"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG,),
            model_types=("zimage", "zimage-turbo"),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
        )

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        """Build engine components."""
        self._device = str(options.get("device", "cuda"))
        self._dtype = str(options.get("dtype", "bf16"))

        runtime = assemble_zimage_runtime(
            spec=ZIMAGE_SPEC,
            codex_components=bundle.components,
            estimated_config=bundle.estimated_config,
            device=self._device,
            dtype=self._dtype,
        )
        self._runtime = runtime
        logger.info("Z Image runtime assembled")

        return CodexObjects(
            unet=runtime.unet,
            clip=None,  # Z Image uses Qwen3, not CLIP
            vae=runtime.vae,
            clipvision=None,
        )

    def _on_unload(self) -> None:
        self._runtime = None

    def _require_runtime(self) -> ZImageEngineRuntime:
        if self._runtime is None:
            raise RuntimeError("Z Image runtime not initialized")
        return self._runtime

    def set_clip_skip(self, clip_skip: int) -> None:
        # Z Image doesn't use CLIP
        pass

    @torch.inference_mode()
    def get_learned_conditioning(self, prompts: list[str]):
        """Encode prompts using Qwen3."""
        runtime = self._require_runtime()
        cond = runtime.text.qwen3_text(prompts)
        return {"crossattn": cond}

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        runtime = self._require_runtime()
        tokens = runtime.text.qwen3_text.tokenize([prompt])
        length = len(tokens[0])
        return length, max(512, length)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        runtime = self._require_runtime()
        memory_management.load_model_gpu(self.codex_objects.vae)
        try:
            return runtime.vae.encode(x)
        finally:
            if self.smart_offload_enabled:
                memory_management.unload_model(self.codex_objects.vae)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        runtime = self._require_runtime()
        memory_management.load_model_gpu(self.codex_objects.vae)
        try:
            return runtime.vae.decode(x)
        finally:
            if self.smart_offload_enabled:
                memory_management.unload_model(self.codex_objects.vae)

    # ------------------------------------------------------------------ Tasks
    def txt2img(self, request: Txt2ImgRequest, **kwargs: Any) -> Iterator[InferenceEvent]:
        """Generate image from text prompt."""
        self.ensure_loaded()
        runtime = self._require_runtime()

        yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing txt2img")

        # Parameters
        prompt = request.prompt
        negative_prompt = getattr(request, "negative_prompt", "") or ""
        width = int(getattr(request, "width", 1024) or 1024)
        height = int(getattr(request, "height", 1024) or 1024)
        steps = int(getattr(request, "steps", ZIMAGE_SPEC.default_steps) or ZIMAGE_SPEC.default_steps)
        cfg_scale = float(getattr(request, "cfg_scale", ZIMAGE_SPEC.default_cfg_scale) or ZIMAGE_SPEC.default_cfg_scale)
        seed = getattr(request, "seed", None)

        logger.info("txt2img: %dx%d, %d steps, cfg=%.1f", width, height, steps, cfg_scale)

        yield ProgressEvent(stage="encoding", percent=0.1, message="Encoding prompt")

        # Text conditioning
        cond = self.get_learned_conditioning([prompt])
        uncond = self.get_learned_conditioning([negative_prompt])

        cond_tensor = cond.get("crossattn")
        uncond_tensor = uncond.get("crossattn")

        yield ProgressEvent(stage="sampling", percent=0.2, message="Starting sampling")

        # Device/dtype
        device = torch.device(self._device if torch.cuda.is_available() else "cpu")
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        dtype = dtype_map.get(self._dtype, torch.bfloat16)

        # Get transformer
        transformer = self.codex_objects.unet.model
        memory_management.load_model_gpu(self.codex_objects.unet)

        try:
            # Sample
            latents = self._sample_flow_matching(
                transformer=transformer,
                cond=cond_tensor.to(device=device, dtype=dtype),
                uncond=uncond_tensor.to(device=device, dtype=dtype),
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                device=device,
                dtype=dtype,
            )

            yield ProgressEvent(stage="decoding", percent=0.9, message="Decoding latents")

            # Decode
            images = self.decode_first_stage(latents)

            # Convert to output format
            images = images.float().clamp(-1, 1) * 0.5 + 0.5
            images = (images * 255).to(torch.uint8)

            # [B, C, H, W] -> list
            output_images = []
            for i in range(images.shape[0]):
                img = images[i].permute(1, 2, 0).cpu().numpy()
                output_images.append(img)

            yield ProgressEvent(stage="complete", percent=1.0, message="Generation complete")
            yield ResultEvent(payload={
                "images": output_images,
                "info": {
                    "engine": self.engine_id,
                    "task": "txt2img",
                    "width": width,
                    "height": height,
                    "steps": steps,
                },
            })

        except Exception as e:
            logger.error("txt2img failed: %s", e)
            yield ProgressEvent(stage="error", percent=1.0, message=str(e))
            yield ResultEvent(payload={
                "images": [],
                "info": {"engine": self.engine_id, "error": str(e)},
            })

    def _sample_flow_matching(
        self,
        transformer,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: Optional[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Flow-matching sampling with CFG."""
        # Latent shape
        latent_h = height // 8
        latent_w = width // 8
        latent_c = 16  # Flux VAE

        B = cond.shape[0]
        shape = (B, latent_c, latent_h, latent_w)

        # Seed
        if seed is not None:
            torch.manual_seed(seed)

        # Initial noise
        x = torch.randn(shape, device=device, dtype=dtype)

        # Sigma schedule (flow-matching)
        shift = ZIMAGE_SPEC.flow_shift
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
        sigmas = shift * timesteps / (1 + (shift - 1) * timesteps)

        # Euler integration
        for i in range(steps):
            t = sigmas[i]
            t_next = sigmas[i + 1]

            timestep = torch.full((B,), float(t), device=device, dtype=dtype)

            # CFG: batch cond and uncond
            x_in = torch.cat([x, x], dim=0)
            cond_in = torch.cat([cond, uncond], dim=0)
            t_in = torch.cat([timestep, timestep], dim=0)

            v_pred = transformer(x_in, t_in, cond_in)

            v_cond, v_uncond = v_pred.chunk(2, dim=0)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)

            # Euler step
            dt = float(t_next) - float(t)
            x = x + dt * v

            if (i + 1) % 2 == 0:
                logger.debug("Step %d/%d: t=%.3f", i + 1, steps, float(t))

        return x
