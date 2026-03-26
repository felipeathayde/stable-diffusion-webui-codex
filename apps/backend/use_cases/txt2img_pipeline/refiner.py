"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Latent model-swap stages for the txt2img pipeline.
Implements the generic first-pass `swap_model` stage plus SDXL refiner stages (global + hires), loading the selected engine, rebuilding
conditioning, and running an additional sampling pass over existing latents through the shared sampler without prompt-control ownership.

Symbols (top-level; keep in sync; no ghosts):
- `SwapModelStage` (dataclass): Executable generic first-pass model-swap stage with selector-authoritative engine loading.
- `GlobalSwapModelStage` (class): First-pass swap-model stage for the global (base) scope.
- `RefinerStage` (dataclass): Executable SDXL refiner stage implementing shared refiner sampling logic and selector-authoritative engine loading.
- `GlobalRefinerStage` (class): Refiner stage for the global (base) scope.
- `HiresRefinerStage` (class): Refiner stage for the hires scope.
"""
# // tags: refiner, pipeline, sdxl, hires, swap_model

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch

from apps.backend.core.engine_loader import EngineLoadOptions, load_engine as _load_engine
from apps.backend.runtime.processing.datatypes import ConditioningPayload, PromptContext
from apps.backend.runtime.processing.models import CodexProcessingTxt2Img, RefinerConfig, SwapStageConfig
from apps.backend.runtime.pipeline_stages.sampling_execute import execute_sampling
from apps.backend.runtime.pipeline_stages.sampling_plan import build_sampling_plan, ensure_sampler_and_rng


RefinerConditioningFn = Callable[[CodexProcessingTxt2Img, PromptContext], tuple[object, object]]
RefinerLogFn = Callable[[object, object], None]
RefinerTensorLogFn = Callable[[str, torch.Tensor | None], None]


@dataclass(slots=True)
class RefinerStage:
    """Executable refiner stage with shared behaviour for global/hires scopes."""

    config: RefinerConfig | None
    label: str

    def is_enabled(self) -> bool:
        cfg = self.config
        return bool(cfg and cfg.enabled and cfg.swap_at_step > 0)

    def run(
        self,
        *,
        processing: CodexProcessingTxt2Img,
        prompt_context: PromptContext,
        noise_settings,
        samples: torch.Tensor,
        compute_conditioning: RefinerConditioningFn,
        log_conditioning: RefinerLogFn,
        log_tensor_stats: RefinerTensorLogFn,
    ) -> torch.Tensor:
        if not self.is_enabled():
            return samples

        cfg = self.config
        assert cfg is not None  # satisfies type-checkers

        selection = cfg.selection
        model_name = selection.require_model_ref(context=self.label)
        if not model_name:
            raise RuntimeError(
                f"{self.label} is enabled but no refiner model was specified. "
                "Provide a valid SDXL refiner checkpoint."
            )

        seed_value = int(cfg.seed)
        if seed_value < 0:
            seed_value = int(torch.randint(0, 2**31 - 1, (1,)).item())

        original_steps = int(processing.steps)
        swap_at_step = int(cfg.swap_at_step)
        if original_steps < 2:
            raise RuntimeError(
                f"{self.label} requires total steps >= 2 for swap semantics (got {original_steps})."
            )
        if swap_at_step < 1 or swap_at_step >= original_steps:
            raise RuntimeError(
                f"{self.label} 'switch_at_step' must be in [1, {original_steps - 1}] (got {swap_at_step})."
            )
        effective_refiner_steps = original_steps - swap_at_step

        logger = logging.getLogger(f"backend.use_cases.txt2img.refiner.{self.label.replace(' ', '_').lower()}")
        logger.info(
            "[refiner] starting %s model=%s swap_at_step=%d remaining_steps=%d cfg=%.3f seed=%d",
            self.label,
            model_name,
            swap_at_step,
            effective_refiner_steps,
            cfg.cfg,
            seed_value,
        )

        load_opts = EngineLoadOptions(
            device=None,
            dtype=None,
            attention_backend=None,
            accelerator=None,
            vae_path=selection.vae_path,
            vae_source=selection.vae_source,
            tenc_path=list(selection.tenc_path) if isinstance(selection.tenc_path, tuple) else selection.tenc_path,
            text_encoder_override=dict(selection.text_encoder_override) if selection.text_encoder_override else None,
            checkpoint_core_only=selection.checkpoint_core_only,
            model_format=selection.model_format,
        )
        try:
            refiner_engine = _load_engine(model_name, options=load_opts)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load SDXL refiner engine '{model_name}': {exc}") from exc

        original_sd_model = processing.sd_model
        original_width = processing.width
        original_height = processing.height
        original_cfg = processing.guidance_scale
        original_cfg_scale = getattr(processing, "cfg_scale", processing.guidance_scale)

        try:
            processing.sd_model = refiner_engine
            latent_h, latent_w = samples.shape[-2], samples.shape[-1]
            processing.width = latent_w * 8
            processing.height = latent_h * 8
            processing.guidance_scale = cfg.cfg
            processing.cfg_scale = cfg.cfg
            processing.steps = original_steps

            plan = build_sampling_plan(
                processing,
                seeds=[seed_value],
                subseeds=[seed_value],
                subseed_strength=0.0,
                noise_settings=noise_settings,
            )
            rng = ensure_sampler_and_rng(processing, plan, latent_channels=samples.shape[1])
            noise = rng.next().to(samples)

            cond_ref, uncond_ref = compute_conditioning(processing, prompt_context)
            if cond_ref is None or uncond_ref is None:
                raise RuntimeError(
                    f"Failed to build conditioning for {self.label.lower()}; get_learned_conditioning returned None."
                )

            payload = ConditioningPayload(conditioning=cond_ref, unconditional=uncond_ref)
            log_conditioning(cond_ref, uncond_ref)

            processing.update_extra_param(
                self.label,
                {
                    "model": model_name,
                    "swap_at_step": int(swap_at_step),
                    "effective_refiner_steps": int(effective_refiner_steps),
                    "total_steps": int(original_steps),
                    "cfg": float(cfg.cfg),
                    "seed": int(seed_value),
                },
            )

            samples_refined = execute_sampling(
                processing,
                plan,
                payload,
                prompt_context,
                prompt_context.loras,
                rng=rng,
                noise=noise,
                init_latent=samples,
                start_at_step=swap_at_step,
                denoise_strength=1.0,
            )
            setattr(processing, "_codex_last_decode_engine", refiner_engine)
            log_tensor_stats(f"{self.label.lower().replace(' ', '_')}_samples", samples_refined)
            return samples_refined
        finally:
            processing.sd_model = original_sd_model
            processing.width = original_width
            processing.height = original_height
            processing.guidance_scale = original_cfg
            processing.cfg_scale = original_cfg_scale
            processing.steps = original_steps


@dataclass(slots=True)
class SwapModelStage:
    """Executable first-pass model-swap stage with persistent engine ownership."""

    config: SwapStageConfig | None
    label: str

    def is_enabled(self) -> bool:
        cfg = self.config
        return bool(cfg and cfg.enabled and cfg.swap_at_step > 0)

    def run(
        self,
        *,
        processing: CodexProcessingTxt2Img,
        prompt_context: PromptContext,
        noise_settings,
        samples: torch.Tensor,
        compute_conditioning: RefinerConditioningFn,
        log_conditioning: RefinerLogFn,
        log_tensor_stats: RefinerTensorLogFn,
    ) -> torch.Tensor:
        if not self.is_enabled():
            return samples

        cfg = self.config
        assert cfg is not None  # satisfies type-checkers

        selection = cfg.selection
        model_name = selection.require_model_ref(context=self.label)
        if not model_name:
            raise RuntimeError(
                f"{self.label} is enabled but no swap model was specified. "
                "Provide a valid checkpoint for the first-pass swap stage."
            )

        seed_value = int(cfg.seed)
        if seed_value < 0:
            seed_value = int(torch.randint(0, 2**31 - 1, (1,)).item())

        original_steps = int(processing.steps)
        swap_at_step = int(cfg.swap_at_step)
        if original_steps < 2:
            raise RuntimeError(
                f"{self.label} requires total steps >= 2 for swap semantics (got {original_steps})."
            )
        if swap_at_step < 1 or swap_at_step >= original_steps:
            raise RuntimeError(
                f"{self.label} 'switch_at_step' must be in [1, {original_steps - 1}] (got {swap_at_step})."
            )
        remaining_steps = original_steps - swap_at_step

        logger = logging.getLogger(f"backend.use_cases.txt2img.swap_model.{self.label.replace(' ', '_').lower()}")
        logger.info(
            "[swap_model] starting %s model=%s swap_at_step=%d remaining_steps=%d cfg=%.3f seed=%d",
            self.label,
            model_name,
            swap_at_step,
            remaining_steps,
            cfg.cfg,
            seed_value,
        )

        load_opts = EngineLoadOptions(
            device=None,
            dtype=None,
            attention_backend=None,
            accelerator=None,
            vae_path=selection.vae_path,
            vae_source=selection.vae_source,
            tenc_path=list(selection.tenc_path) if isinstance(selection.tenc_path, tuple) else selection.tenc_path,
            text_encoder_override=dict(selection.text_encoder_override) if selection.text_encoder_override else None,
            checkpoint_core_only=selection.checkpoint_core_only,
            model_format=selection.model_format,
            zimage_variant=selection.zimage_variant,
        )
        try:
            swap_engine = _load_engine(model_name, options=load_opts)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load swap_model engine '{model_name}': {exc}") from exc

        original_sd_model = processing.sd_model
        original_width = processing.width
        original_height = processing.height
        original_cfg = processing.guidance_scale
        original_cfg_scale = getattr(processing, "cfg_scale", processing.guidance_scale)

        try:
            processing.sd_model = swap_engine
            latent_h, latent_w = samples.shape[-2], samples.shape[-1]
            processing.width = latent_w * 8
            processing.height = latent_h * 8
            processing.guidance_scale = cfg.cfg
            processing.cfg_scale = cfg.cfg
            processing.steps = original_steps

            plan = build_sampling_plan(
                processing,
                seeds=[seed_value],
                subseeds=[seed_value],
                subseed_strength=0.0,
                noise_settings=noise_settings,
            )
            rng = ensure_sampler_and_rng(processing, plan, latent_channels=samples.shape[1])
            noise = rng.next().to(samples)

            cond_swap, uncond_swap = compute_conditioning(processing, prompt_context)
            if cond_swap is None or uncond_swap is None:
                raise RuntimeError(
                    f"Failed to build conditioning for {self.label.lower()}; get_learned_conditioning returned None."
                )

            payload = ConditioningPayload(conditioning=cond_swap, unconditional=uncond_swap)
            log_conditioning(cond_swap, uncond_swap)

            processing.update_extra_param(
                self.label,
                {
                    "model": model_name,
                    "swap_at_step": int(swap_at_step),
                    "remaining_steps": int(remaining_steps),
                    "total_steps": int(original_steps),
                    "cfg": float(cfg.cfg),
                    "seed": int(seed_value),
                },
            )

            swapped_samples = execute_sampling(
                processing,
                plan,
                payload,
                prompt_context,
                prompt_context.loras,
                rng=rng,
                noise=noise,
                init_latent=samples,
                start_at_step=swap_at_step,
                denoise_strength=1.0,
            )
            setattr(processing, "_codex_last_decode_engine", swap_engine)
            log_tensor_stats(f"{self.label.lower().replace(' ', '_')}_samples", swapped_samples)
            return swapped_samples
        finally:
            processing.sd_model = swap_engine
            processing.width = original_width
            processing.height = original_height
            processing.guidance_scale = original_cfg
            processing.cfg_scale = original_cfg_scale
            processing.steps = original_steps


class GlobalSwapModelStage(SwapModelStage):
    def __init__(self, config: SwapStageConfig | None):
        super().__init__(config=config, label="Swap Model")


class GlobalRefinerStage(RefinerStage):
    def __init__(self, config: RefinerConfig | None):
        super().__init__(config=config, label="Refiner")


class HiresRefinerStage(RefinerStage):
    def __init__(self, config: RefinerConfig | None):
        super().__init__(config=config, label="Hires Refiner")
