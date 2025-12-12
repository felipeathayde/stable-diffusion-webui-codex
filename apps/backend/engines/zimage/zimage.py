"""Z Image Turbo Engine.

Alibaba Z Image Turbo (6B) txt2img engine following Flux pattern.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models.loader import DiffusionModelBundle
from apps.backend.runtime.zimage.debug import env_flag, env_int, truncate_text

from .spec import ZIMAGE_SPEC, ZImageEngineRuntime, assemble_zimage_runtime

logger = logging.getLogger("backend.engines.zimage.zimage")


class _ZImagePromptList(list[str]):
    """List-like prompt wrapper used to carry per-run metadata."""

    def __init__(self, items: list[str], *, distilled_cfg_scale: float, is_negative_prompt: bool) -> None:
        super().__init__(items)
        self.distilled_cfg_scale = float(distilled_cfg_scale)
        self.is_negative_prompt = bool(is_negative_prompt)


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
            external_vae_path=options.get("vae_path"),
            external_tenc_path=options.get("tenc_path"),
        )
        logger.info(
            "Z Image build: vae_path=%s tenc_path=%s all_options=%s",
            options.get("vae_path"),
            options.get("tenc_path"),
            list(options.keys()),
        )
        self._runtime = runtime
        logger.info("Z Image runtime assembled")

        # Turbo models use distilled guidance; disable CFG/uncond conditioning.
        self.use_distilled_cfg_scale = True

        return CodexObjects(
            unet=runtime.unet,
            vae=runtime.vae,
            text_encoders={"qwen3": runtime.text.qwen3_text},
            clipvision=None,
        )

    @property
    def required_text_encoders(self) -> tuple[str, ...]:
        """Z Image uses Qwen3 text encoder, not CLIP."""
        return ("qwen3",)

    def _on_unload(self) -> None:
        self._runtime = None

    def _require_runtime(self) -> ZImageEngineRuntime:
        if self._runtime is None:
            raise RuntimeError("Z Image runtime not initialized")
        return self._runtime

    def set_clip_skip(self, clip_skip: int) -> None:
        # Z Image doesn't use CLIP
        pass

    def _prepare_prompt_wrappers(
        self,
        texts: list[str],
        proc: Any,
        *,
        is_negative: bool,
    ) -> _ZImagePromptList:
        distilled = getattr(proc, "distilled_guidance_scale", None)
        if distilled is None:
            distilled = getattr(proc, "distilled_cfg", None)
        try:
            distilled_value = float(distilled) if distilled is not None else float(ZIMAGE_SPEC.default_cfg_scale)
        except Exception:
            distilled_value = float(ZIMAGE_SPEC.default_cfg_scale)
        return _ZImagePromptList([str(t or "") for t in texts], distilled_cfg_scale=distilled_value, is_negative_prompt=is_negative)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompts: list[str]):
        """Encode prompts using Qwen3."""
        runtime = self._require_runtime()
        
        # Load text encoder to GPU using memory management (same pattern as Flux)
        memory_management.load_model_gpu(runtime.clip.patcher)
        unload_clip = self.smart_offload_enabled
        
        # Lumina 2 / Z Image conditioning format (historical; keep for now).
        # Note: tokenizer applies the Qwen3 chat template separately unless the prompt
        # already begins with '<|im_start|>'.
        system_prompt = (
            "You are an assistant designed to generate superior images with the superior degree of image-text alignment "
            "based on textual prompts or user prompts."
        )
        formatted_prompts = [f"{system_prompt} <Prompt Start> {p}" for p in prompts]
        if env_flag("CODEX_ZIMAGE_DEBUG_PROMPT", False) and formatted_prompts:
            logger.info(
                "[zimage-debug] prompt0=%s",
                truncate_text(formatted_prompts[0], limit=env_int("CODEX_ZIMAGE_DEBUG_TEXT_MAX", 400)),
            )
        
        try:
            cond = runtime.text.qwen3_text(formatted_prompts)
        finally:
            if unload_clip:
                memory_management.unload_model(runtime.clip.patcher)
        
        # Z Image uses Qwen3 which doesn't have pooled output like CLIP.
        # Provide a zeros placeholder for 'vector' to satisfy compile_conditions.
        batch_size = len(prompts)
        # Use a small vector dimension - this is just a placeholder
        vector = torch.zeros(batch_size, 768, dtype=cond.dtype, device=cond.device)
        
        # Distilled CFG guidance (Z Image uses distilled guidance like Flux)
        distilled_cfg = getattr(prompts, "distilled_cfg_scale", ZIMAGE_SPEC.default_cfg_scale) or ZIMAGE_SPEC.default_cfg_scale
        guidance = torch.full((batch_size,), float(distilled_cfg), dtype=torch.float32)
        if env_flag("CODEX_ZIMAGE_DEBUG_PROMPT", False):
            logger.info("[zimage-debug] distilled_cfg_scale=%.3f", float(distilled_cfg))
        
        return {
            "crossattn": cond,
            "vector": vector,
            "guidance": guidance,
        }

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
            # Match Flux/Z-Image Flow16 VAE semantics:
            # - VAE wrapper expects pixel samples as BHWC in [0, 1]
            # - Latents used by the flow core must be normalized via process_in()
            sample = runtime.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
            sample = runtime.vae.first_stage_model.process_in(sample)
            return sample.to(x)
        finally:
            if self.smart_offload_enabled:
                memory_management.unload_model(self.codex_objects.vae)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        runtime = self._require_runtime()
        memory_management.load_model_gpu(self.codex_objects.vae)
        try:
            # Match Flux/Z-Image Flow16 VAE semantics:
            # - Model operates in normalized latent space
            # - VAE decode expects denormalized latents via process_out()
            sample = runtime.vae.first_stage_model.process_out(x)
            sample = runtime.vae.decode(sample)
            return sample.to(x)
        finally:
            if self.smart_offload_enabled:
                memory_management.unload_model(self.codex_objects.vae)
