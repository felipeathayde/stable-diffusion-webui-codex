"""WAN 2.2 14B Engine (Clean implementation matching Flux pattern).

This is a complete rewrite using the WanTransformer2DModel and
centralized runtime assembly pattern.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent, Txt2VidRequest, Img2VidRequest
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models.loader import DiffusionModelBundle, resolve_diffusion_bundle

from .spec import WAN_14B_SPEC, WanEngineRuntime, assemble_wan_runtime

logger = logging.getLogger("backend.engines.wan22.wan22_14b")


class Wan2214BEngine(CodexDiffusionEngine):
    """Codex native WAN 2.2 14B engine (matching Flux pattern)."""

    engine_id = "wan22_14b"

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[WanEngineRuntime] = None
        self._streaming_controller = None
        self._device = "cuda"
        self._dtype = "bf16"

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2VID, TaskType.IMG2VID),
            model_types=("wan-2.2-14b",),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
        )

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        """Build engine components using centralized runtime assembly."""
        self._device = str(options.get("device", "cuda"))
        self._dtype = str(options.get("dtype", "bf16"))

        runtime = assemble_wan_runtime(
            spec=WAN_14B_SPEC,
            codex_components=bundle.components,
            estimated_config=bundle.estimated_config,
            device=self._device,
            dtype=self._dtype,
        )
        self._runtime = runtime
        logger.info("WAN runtime assembled for %s", WAN_14B_SPEC.name)

        # Streaming configuration
        streaming_enabled = options.get("core_streaming_enabled", False)
        if streaming_enabled:
            streaming_policy = options.get("core_streaming_policy", "naive")
            blocks_per_segment = options.get("core_streaming_blocks_per_segment", 4)
            logger.info(
                "WAN core streaming enabled: policy=%s, blocks_per_segment=%d",
                streaming_policy,
                blocks_per_segment,
            )
            try:
                from apps.backend.runtime.wan22.streaming import (
                    build_execution_plan,
                    WanCoreController,
                    WanStreamingPolicy,
                    StreamedWanTransformer,
                )
                # Get transformer from patcher
                core_model = runtime.unet.model
                plan = build_execution_plan(core_model, blocks_per_segment=blocks_per_segment)
                controller = WanCoreController(
                    storage_device="cpu",
                    compute_device="cuda" if torch.cuda.is_available() else "cpu",
                    policy=WanStreamingPolicy(streaming_policy),
                )
                streamed_core = StreamedWanTransformer(core_model, plan, controller)
                runtime.unet.model = streamed_core
                self._streaming_controller = controller
                logger.info(
                    "WAN streaming active: %d segments, %.2f MB total",
                    len(plan),
                    plan.total_bytes / (1024 * 1024),
                )
            except Exception as e:
                logger.warning("Failed to enable WAN streaming: %s", e)
                self._streaming_controller = None
        else:
            self._streaming_controller = None

        return CodexObjects(
            unet=runtime.unet,
            clip=None,  # WAN uses T5 only, accessed via runtime.text
            vae=runtime.vae,
            clipvision=None,
        )

    def _on_unload(self) -> None:
        self._runtime = None
        self._streaming_controller = None

    def _require_runtime(self) -> WanEngineRuntime:
        if self._runtime is None:
            raise RuntimeError("WAN runtime not initialized; call load() first.")
        return self._runtime

    def set_clip_skip(self, clip_skip: int) -> None:
        # WAN doesn't use CLIP, this is a no-op
        pass

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        """Encode text prompts using T5."""
        runtime = self._require_runtime()
        # Load T5 to GPU
        # TODO: Wire proper T5 loading
        cond_t5 = runtime.text.t5_text(prompt)
        return {"crossattn": cond_t5}

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        runtime = self._require_runtime()
        token_count = len(runtime.text.t5_text.tokenize([prompt])[0])
        return token_count, max(256, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video frames to latents."""
        runtime = self._require_runtime()
        memory_management.load_model_gpu(self.codex_objects.vae)
        unload = self.smart_offload_enabled
        try:
            # WAN VAE expects [B, C, T, H, W]
            sample = runtime.vae.encode(x)
            return sample.to(x)
        finally:
            if unload:
                memory_management.unload_model(self.codex_objects.vae)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latents to video frames."""
        runtime = self._require_runtime()
        memory_management.load_model_gpu(self.codex_objects.vae)
        unload = self.smart_offload_enabled
        try:
            sample = runtime.vae.decode(x)
            return sample.to(x)
        finally:
            if unload:
                memory_management.unload_model(self.codex_objects.vae)

    # ------------------------------------------------------------------ Tasks
    def txt2vid(self, request: Txt2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:
        """Generate video from text prompt."""
        self.ensure_loaded()
        runtime = self._require_runtime()

        yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing txt2vid")

        # Get parameters from request
        prompt = request.prompt
        negative_prompt = getattr(request, "negative_prompt", "") or ""
        width = int(getattr(request, "width", 768) or 768)
        height = int(getattr(request, "height", 432) or 432)
        num_frames = int(getattr(request, "num_frames", 16) or 16)
        steps = int(getattr(request, "steps", WAN_14B_SPEC.default_steps) or WAN_14B_SPEC.default_steps)
        cfg_scale = float(getattr(request, "cfg_scale", WAN_14B_SPEC.default_cfg_scale) or WAN_14B_SPEC.default_cfg_scale)
        seed = getattr(request, "seed", None)

        logger.info(
            "txt2vid: %dx%d, %d frames, %d steps, cfg=%.1f",
            width, height, num_frames, steps, cfg_scale,
        )

        yield ProgressEvent(stage="encoding", percent=0.1, message="Encoding prompt")

        # Text conditioning
        cond = self.get_learned_conditioning([prompt])
        uncond = self.get_learned_conditioning([negative_prompt])
        
        # Get conditioning tensors
        cond_tensor = cond.get("crossattn") if isinstance(cond, dict) else cond
        uncond_tensor = uncond.get("crossattn") if isinstance(uncond, dict) else uncond

        yield ProgressEvent(stage="sampling", percent=0.2, message="Starting sampling")

        # Import sampler
        from apps.backend.runtime.wan22.sampler import sample_txt2vid
        
        # Resolve device and dtype
        device = torch.device(self._device if torch.cuda.is_available() or self._device == "cpu" else "cpu")
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        dtype = dtype_map.get(self._dtype, torch.bfloat16)
        
        # Get transformer from unet patcher
        transformer = self.codex_objects.unet.model
        vae = runtime.vae
        
        # Load models to GPU
        memory_management.load_model_gpu(self.codex_objects.unet)
        
        # Progress tracking for yield
        progress_events = []
        
        def sampling_callback(step: int, total: int, latent: torch.Tensor):
            pct = 0.2 + 0.7 * (step / total)  # 20% to 90%
            progress_events.append(ProgressEvent(
                stage="sampling",
                percent=pct,
                step=step,
                total_steps=total,
                message=f"Sampling step {step}/{total}",
            ))
        
        try:
            # Run sampling
            video = sample_txt2vid(
                transformer=transformer,
                vae=vae,
                cond=cond_tensor.to(device=device, dtype=dtype),
                uncond=uncond_tensor.to(device=device, dtype=dtype) if uncond_tensor is not None else None,
                width=width,
                height=height,
                num_frames=num_frames,
                num_steps=steps,
                cfg_scale=cfg_scale,
                flow_shift=WAN_14B_SPEC.flow_shift,
                seed=seed,
                device=device,
                dtype=dtype,
                callback=sampling_callback,
            )
            
            # Yield accumulated progress events
            for evt in progress_events:
                yield evt
            
            yield ProgressEvent(stage="decoding", percent=0.9, message="Processing output")
            
            # Convert video tensor to frame list
            # video shape: [B, C, T, H, W] -> list of PIL images
            frames = []
            if video is not None and video.numel() > 0:
                # Normalize to 0-255 and convert
                video = video.float().clamp(-1, 1) * 0.5 + 0.5  # [-1,1] -> [0,1]
                video = (video * 255).to(torch.uint8)
                
                # video: [B, C, T, H, W] -> iterate over T
                B, C, T, H, W = video.shape
                for t in range(T):
                    frame = video[0, :, t, :, :].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                    frames.append(frame)
            
            yield ProgressEvent(stage="complete", percent=1.0, message="Generation complete")
            yield ResultEvent(payload={
                "images": frames,
                "info": {
                    "engine": self.engine_id,
                    "task": "txt2vid",
                    "frames": len(frames),
                    "width": width,
                    "height": height,
                    "steps": steps,
                },
            })
            
        except Exception as e:
            logger.error("txt2vid failed: %s", e)
            yield ProgressEvent(stage="error", percent=1.0, message=str(e))
            yield ResultEvent(payload={
                "images": [],
                "info": {"engine": self.engine_id, "task": "txt2vid", "error": str(e)},
            })

    def img2vid(self, request: Img2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:
        """Generate video from image."""
        self.ensure_loaded()
        runtime = self._require_runtime()

        if getattr(request, "init_image", None) is None:
            raise RuntimeError("img2vid requires 'init_image'")

        yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing img2vid")

        # TODO: Implement img2vid pipeline
        logger.warning("img2vid not fully implemented yet")

        yield ProgressEvent(stage="complete", percent=1.0, message="Generation complete")
        yield ResultEvent(payload={
            "images": [],
            "info": {"engine": self.engine_id, "task": "img2vid", "status": "placeholder"},
        })
