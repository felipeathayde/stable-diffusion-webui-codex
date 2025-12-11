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

    @torch.inference_mode()
    def get_learned_conditioning(self, prompts: list[str]):
        """Encode prompts using Qwen3."""
        runtime = self._require_runtime()
        
        # Load text encoder to GPU using memory management (same pattern as Flux)
        memory_management.load_model_gpu(runtime.clip.patcher)
        unload_clip = self.smart_offload_enabled
        
        # Lumina 2 / Z Image conditioning format
        # System prompt + <Prompt Start> + User Prompt
        system_prompt = "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts."
        formatted_prompts = [f"{system_prompt} <Prompt Start> {p}" for p in prompts]
        
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
            return runtime.vae.encode(x)
        finally:
            if self.smart_offload_enabled:
                memory_management.unload_model(self.codex_objects.vae)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        runtime = self._require_runtime()
        memory_management.load_model_gpu(self.codex_objects.vae)
        try:
            # VAE returns BCHW in [-1, 1] range directly
            return runtime.vae.decode(x)
        finally:
            if self.smart_offload_enabled:
                memory_management.unload_model(self.codex_objects.vae)


