from __future__ import annotations

import logging
from typing import Any, Iterable, List, Mapping, Optional

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.flux.spec import FLUX_SPEC, FluxEngineRuntime, assemble_flux_runtime
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models.loader import DiffusionModelBundle

logger = logging.getLogger("backend.engines.flux")


class Flux(CodexDiffusionEngine):
    """Codex native Flux engine."""

    engine_id = "flux"

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[FluxEngineRuntime] = None
        self._guidance_default = FLUX_SPEC.distilled_cfg_scale_default

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG, TaskType.IMG2IMG),
            model_types=("flux",),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
        )

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        runtime = assemble_flux_runtime(
            spec=FLUX_SPEC,
            estimated_config=bundle.estimated_config,
            codex_components=bundle.components,
        )
        self._runtime = runtime
        self.use_distilled_cfg_scale = runtime.use_distilled_cfg
        logger.debug("Flux runtime prepared (distilled cfg=%s)", runtime.use_distilled_cfg)

        # Streaming configuration from options
        streaming_enabled = options.get("core_streaming_enabled", False)
        if streaming_enabled:
            streaming_policy = options.get("core_streaming_policy", "naive")
            blocks_per_segment = options.get("core_streaming_blocks_per_segment", 4)
            logger.info(
                "Flux core streaming enabled: policy=%s, blocks_per_segment=%d",
                streaming_policy,
                blocks_per_segment,
            )
            try:
                from apps.backend.runtime.flux.streaming import (
                    trace_execution_plan,
                    CoreController,
                    StreamedFluxCore,
                    StreamingPolicy,
                )
                # Get the actual FluxTransformer2DModel from the patcher
                core_model = runtime.unet.model
                plan = trace_execution_plan(core_model, blocks_per_segment=blocks_per_segment)
                controller = CoreController(
                    storage_device="cpu",
                    compute_device="cuda" if torch.cuda.is_available() else "cpu",
                    policy=StreamingPolicy(streaming_policy),
                )
                streamed_core = StreamedFluxCore(core_model, plan, controller)
                # Replace the model in the patcher
                runtime.unet.model = streamed_core
                self._streaming_controller = controller
                logger.info(
                    "Flux streaming active: %d segments, %.2f MB total",
                    len(plan),
                    plan.total_bytes / (1024 * 1024),
                )
            except Exception as e:
                logger.warning("Failed to enable Flux streaming: %s", e)
                self._streaming_controller = None
        else:
            self._streaming_controller = None

        return CodexObjects(
            unet=runtime.unet,
            clip=runtime.clip,
            vae=runtime.vae,
            clipvision=None,
        )

    def _on_unload(self) -> None:
        self._runtime = None

    def _require_runtime(self) -> FluxEngineRuntime:
        if self._runtime is None:
            raise RuntimeError("Flux runtime is not initialised; call load() first.")
        return self._runtime

    def set_clip_skip(self, clip_skip: int) -> None:
        runtime = self._require_runtime()
        runtime.set_clip_skip(clip_skip)
        logger.debug("Flux clip skip set to %d", clip_skip)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: List[str]):
        runtime = self._require_runtime()
        memory_management.load_model_gpu(self.codex_objects.clip.patcher)
        unload_clip = self.smart_offload_enabled
        try:
            clip_branch = runtime.text.clip_text
            cond_l, pooled_l = (clip_branch(prompt) if clip_branch is not None else (None, None))
            cond_t5 = runtime.text.t5_text(prompt)
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
        finally:
            if unload_clip:
                memory_management.unload_model(self.codex_objects.clip.patcher)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        token_count = len(runtime.text.t5_text.tokenize([prompt])[0])
        return token_count, max(255, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        memory_management.load_model_gpu(self.codex_objects.vae)
        unload_vae = self.smart_offload_enabled
        try:
            sample = self.codex_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
            sample = self.codex_objects.vae.first_stage_model.process_in(sample)
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.unload_model(self.codex_objects.vae)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        memory_management.load_model_gpu(self.codex_objects.vae)
        unload_vae = self.smart_offload_enabled
        try:
            sample = self.codex_objects.vae.first_stage_model.process_out(x)
            sample = self.codex_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.unload_model(self.codex_objects.vae)
