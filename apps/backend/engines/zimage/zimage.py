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
from apps.backend.runtime.memory.smart_offload import (
    record_smart_cache_hit,
    record_smart_cache_miss,
    smart_cache_enabled,
)
from apps.backend.runtime.models.loader import DiffusionModelBundle
from apps.backend.runtime.timeline import timeline_node
from apps.backend.runtime.zimage.debug import env_flag, env_int, truncate_text

from .spec import ZIMAGE_SPEC, ZImageEngineRuntime, assemble_zimage_runtime

logger = logging.getLogger("backend.engines.zimage.zimage")


class _ZImagePromptList(list[str]):
    """List-like prompt wrapper used to carry per-run metadata."""

    def __init__(
        self,
        items: list[str],
        *,
        distilled_cfg_scale: float,
        is_negative_prompt: bool,
        smart_cache: bool | None,
    ) -> None:
        super().__init__(items)
        self.distilled_cfg_scale = float(distilled_cfg_scale)
        self.is_negative_prompt = bool(is_negative_prompt)
        self.smart_cache = smart_cache


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
        smart_flag = getattr(proc, "smart_cache", None)
        smart_value = None if smart_flag is None else bool(smart_flag)
        return _ZImagePromptList(
            [str(t or "") for t in texts],
            distilled_cfg_scale=distilled_value,
            is_negative_prompt=is_negative,
            smart_cache=smart_value,
        )

    @timeline_node("text_encoding", "get_learned_conditioning")
    @torch.inference_mode()
    def get_learned_conditioning(self, prompts: list[str]):
        """Encode prompts using Qwen3."""
        runtime = self._require_runtime()

        texts = tuple(str(x or "") for x in prompts)
        is_negative = bool(getattr(prompts, "is_negative_prompt", False))
        smart_flag = getattr(prompts, "smart_cache", None)
        use_cache = bool(smart_flag) if smart_flag is not None else smart_cache_enabled()
        max_length = getattr(runtime.text.qwen3_text, "max_length", None)
        cache_key = (texts, is_negative, max_length)

        if use_cache:
            cached = self._cond_cache.get(cache_key)
            if isinstance(cached, torch.Tensor):
                record_smart_cache_hit("zimage.conditioning")
                target_device = memory_management.text_encoder_device()
                return cached.to(target_device)
            record_smart_cache_miss("zimage.conditioning")

        # Load text encoder to GPU using memory management (same pattern as Flux)
        memory_management.load_model_gpu(runtime.clip.patcher)
        unload_clip = self.smart_offload_enabled
        
        # Per diffusers reference: pass prompts directly - the tokenizer's
        # apply_chat_template with enable_thinking=True handles formatting.
        # Do NOT add manual system prompt or <Prompt Start> markers.
        if env_flag("CODEX_ZIMAGE_DEBUG_PROMPT", False) and prompts:
            logger.info(
                "[zimage-debug] prompt0=%s",
                truncate_text(prompts[0], limit=env_int("CODEX_ZIMAGE_DEBUG_TEXT_MAX", 400)),
            )
        
        try:
            cond = runtime.text.qwen3_text(prompts)
            if use_cache:
                # Keep cache bounded: store only the most recent entry (tensors on CPU).
                self._cond_cache.clear()
                self._cond_cache[cache_key] = cond.detach().to("cpu")
        finally:
            if unload_clip:
                memory_management.unload_model(runtime.clip.patcher)

        raw_cfg = getattr(prompts, "distilled_cfg_scale", None)
        distilled_cfg = float(raw_cfg) if raw_cfg is not None else float(ZIMAGE_SPEC.default_cfg_scale)
        if env_flag("CODEX_ZIMAGE_DEBUG_PROMPT", False):
            logger.info("[zimage-debug] distilled_cfg_scale=%.3f", distilled_cfg)

        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        runtime = self._require_runtime()
        tokens = runtime.text.qwen3_text.tokenize([prompt])
        length = len(tokens[0])
        return length, max(512, length)

    @timeline_node("vae", "encode_first_stage")
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

    @timeline_node("vae", "decode_first_stage")
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

    @torch.inference_mode()
    def sample_with_diffusers(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
    ) -> list:
        """Run generation using Diffusers ZImagePipeline directly.
        
        This bypasses all Codex sampling and uses Diffusers scheduler exactly
        as in the reference implementation. For debugging/validation.
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt for CFG (typically None for Turbo)
            height: Image height
            width: Image width  
            num_inference_steps: Sampling steps (9 for Turbo)
            guidance_scale: CFG scale (0.0 for Turbo)
            seed: Random seed
        
        Returns:
            List of PIL images
        """
        from .standalone_sampler import sample_zimage_diffusers_math, decode_latents
        from PIL import Image
        import numpy as np
        
        runtime = self._require_runtime()
        
        logger.info("[zimage] Running standalone Diffusers-math sampler")
        
        # Step 1: Encode prompt using OUR working text encoder
        # This already works correctly and handles all the Qwen3 specifics
        prompts_list = [prompt] if isinstance(prompt, str) else list(prompt)
        cond = self.get_learned_conditioning(prompts_list)
        text_embeddings = cond["crossattn"] if isinstance(cond, dict) else cond  # [B, seq, hidden]
        
        logger.info("[zimage] text_embeddings: shape=%s dtype=%s", text_embeddings.shape, text_embeddings.dtype)
        
        # Step 2: Get transformer (raw model, not wrapped)
        transformer_model = runtime.unet.model.diffusion_model
        
        # Load transformer to GPU
        memory_management.load_model_gpu(runtime.unet)
        
        try:
            # Step 3: Sample using Diffusers scheduler + negation
            computation_dtype = torch.bfloat16 if self._dtype == "bf16" else torch.float16
            
            if seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(seed)
            else:
                generator = None
            
            latents = sample_zimage_diffusers_math(
                transformer=transformer_model,
                text_embeddings=text_embeddings,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                device=self._device,
                dtype=computation_dtype,
            )
            
            logger.info("[zimage] sampling done, latents: shape=%s dtype=%s", latents.shape, latents.dtype)
            
        finally:
            if self.smart_offload_enabled:
                memory_management.unload_model(runtime.unet)
        
        # Step 4: Decode latents to images
        memory_management.load_model_gpu(self.codex_objects.vae)
        try:
            images_tensor = decode_latents(runtime.vae, latents)
            
            # Convert to PIL images
            images = []
            for i in range(images_tensor.shape[0]):
                img_np = images_tensor[i].permute(1, 2, 0).cpu().float().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                images.append(Image.fromarray(img_np))
            
            logger.info("[zimage] decoded %d images", len(images))
            return images
            
        finally:
            if self.smart_offload_enabled:
                memory_management.unload_model(self.codex_objects.vae)
