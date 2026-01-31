"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 14B engine (Codex runtime assembly) for txt2vid.
Resolves the model bundle, assembles a `WanEngineRuntime` via `CodexWan22Factory`, and executes video requests with optional core
streaming settings (Flux-like engine pattern).

Symbols (top-level; keep in sync; no ghosts):
- `Wan2214BEngine` (class): Codex diffusion engine for WAN22 14B; assembles runtime and handles txt2vid runs (img2vid is not yet ported).
  (contains nested helpers for core streaming and bundle resolution).
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Mapping, Optional

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent, Txt2VidRequest
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.common.runtime_lifecycle import require_runtime
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models.loader import DiffusionModelBundle

from .factory import CodexWan22Factory
from .spec import WAN_14B_SPEC, WanEngineRuntime

logger = logging.getLogger("backend.engines.wan22.wan22_14b")

_WAN14B_FACTORY = CodexWan22Factory(spec=WAN_14B_SPEC)


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
            tasks=(TaskType.TXT2VID,),
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
        assembly = _WAN14B_FACTORY.assemble(bundle, options=options)
        runtime = assembly.runtime
        self._runtime = runtime
        self._device = str(getattr(runtime, "device", "cuda"))
        self._dtype = str(getattr(runtime, "dtype", "bf16"))
        logger.info("WAN runtime assembled for %s", WAN_14B_SPEC.name)

        # Streaming configuration (fail loud when explicitly requested).
        streaming_enabled = bool(options.get("core_streaming_enabled", False))
        self._streaming_controller = None
        if streaming_enabled:
            streaming_policy = str(options.get("core_streaming_policy", "naive"))
            blocks_raw = options.get("core_streaming_blocks_per_segment", 4)
            try:
                blocks_per_segment = int(blocks_raw)
            except Exception as exc:  # noqa: BLE001
                raise TypeError("core_streaming_blocks_per_segment must be an integer") from exc
            if blocks_per_segment < 1:
                raise ValueError("core_streaming_blocks_per_segment must be >= 1")
            logger.info(
                "WAN core streaming enabled: policy=%s, blocks_per_segment=%d",
                streaming_policy,
                blocks_per_segment,
            )

            from apps.backend.runtime.families.wan22.streaming import (
                StreamedWanTransformer,
                WanCoreController,
                WanStreamingPolicy,
                build_execution_plan,
            )

            core_model = getattr(runtime.denoiser.model, "diffusion_model", None)
            if core_model is None:
                raise RuntimeError("WAN denoiser wrapper does not expose diffusion_model; cannot enable streaming.")

            try:
                plan = build_execution_plan(core_model, blocks_per_segment=blocks_per_segment)
                controller = WanCoreController(
                    storage_device="cpu",
                    compute_device="cuda" if torch.cuda.is_available() else "cpu",
                    policy=WanStreamingPolicy(streaming_policy),
                )
                streamed_core = StreamedWanTransformer(core_model, plan, controller)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"WAN core streaming requested but failed to enable: {exc}") from exc

            runtime.denoiser.model.diffusion_model = streamed_core
            self._streaming_controller = controller
            logger.info(
                "WAN streaming active: %d segments, %.2f MB total",
                len(plan),
                plan.total_bytes / (1024 * 1024),
            )

        return assembly.codex_objects

    @property
    def required_text_encoders(self) -> tuple[str, ...]:
        """WAN22 uses T5 text encoder only, not CLIP."""
        return ("t5",)

    def _on_unload(self) -> None:
        self._runtime = None
        self._streaming_controller = None

    def _require_runtime(self) -> WanEngineRuntime:
        return require_runtime(self._runtime, label=self.engine_id)

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
        min_len = int(getattr(runtime.text.t5_text, "min_length", 256) or 256)
        return token_count, max(min_len, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video frames to latents."""
        runtime = self._require_runtime()
        memory_management.manager.load_model(self.codex_objects.vae)
        unload = self.smart_offload_enabled
        try:
            # WAN VAE expects [B, C, T, H, W]
            sample = runtime.vae.encode(x)
            return sample.to(x)
        finally:
            if unload:
                memory_management.manager.unload_model(self.codex_objects.vae)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latents to video frames."""
        runtime = self._require_runtime()
        memory_management.manager.load_model(self.codex_objects.vae)
        unload = self.smart_offload_enabled
        try:
            sample = runtime.vae.decode(x)
            return sample.to(x)
        finally:
            if unload:
                memory_management.manager.unload_model(self.codex_objects.vae)

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
        req_cfg = getattr(request, "guidance_scale", None)
        cfg_scale = float(req_cfg if req_cfg is not None else WAN_14B_SPEC.default_cfg_scale)
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
        from apps.backend.runtime.families.wan22.sampler import sample_txt2vid
        
        # Resolve device and dtype
        device = torch.device(self._device if torch.cuda.is_available() or self._device == "cpu" else "cpu")
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        dtype = dtype_map.get(self._dtype, torch.bfloat16)
        
        # Get transformer core from the KModel wrapper
        transformer = getattr(self.codex_objects.denoiser.model, "diffusion_model", self.codex_objects.denoiser.model)
        vae = runtime.vae
        
        # Load models to GPU
        memory_management.manager.load_model(self.codex_objects.denoiser)
        
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

    def img2vid(self, request: Any, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        raise NotImplementedError("wan22_14b img2vid not yet ported")
