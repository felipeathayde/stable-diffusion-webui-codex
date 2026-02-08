"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampling execution helper for pipeline orchestrators.
Runs the sampler loop, integrates preview callbacks, applies LoRAs, and triggers post-sample hooks and diagnostics (including ER-SDE option
propagation into the sampler and diagnostic metadata dumps).

Symbols (top-level; keep in sync; no ghosts):
- `_maybe_dump_latents` (function): Dump latents to disk when enabled via env flags (debug diagnostics + effective ER-SDE metadata).
- `execute_sampling` (function): Execute sampling given processing + plan + conditioning payload and return the sampled latents.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from apps.backend.core import devices
from apps.backend.core.rng import ImageRNG
from apps.backend.core.state import state as backend_state
from apps.backend.patchers.lora_apply import apply_loras_to_engine
from apps.backend.runtime.live_preview import (
    LivePreviewMethod,
    decode_preview_image,
    debug_preview_factors_enabled,
    live_preview_method,
    maybe_log_preview_factors,
)
from apps.backend.runtime.processing.conditioners import txt2img_conditioning
from apps.backend.runtime.processing.datatypes import ConditioningPayload, PromptContext, SamplingPlan
from apps.backend.runtime.sampling.context import build_sampling_context
from apps.backend.infra.config.env_flags import env_flag
from apps.backend.infra.config.repo_root import get_repo_root

from .scripts import collect_lora_selections, run_before_sampling_hooks, run_post_sample_hooks

logger = logging.getLogger(__name__)


def _maybe_dump_latents(
    latents: torch.Tensor,
    processing: Any,
    plan: SamplingPlan,
    prompt_context: PromptContext,
) -> None:
    if not env_flag("CODEX_DUMP_LATENTS", default=False):
        return

    path_hint = os.getenv("CODEX_DUMP_LATENTS_PATH")
    if path_hint:
        target = Path(path_hint).expanduser()
    else:
        target = get_repo_root() / "logs" / "diagnostics"
    if not target.suffix:
        timestamp = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        target = target / f"latents-{timestamp}.pt"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "latents": latents.detach().cpu(),
            "metadata": {
                "timestamp_utc": _dt.datetime.utcnow().isoformat(timespec="seconds"),
                "width": int(getattr(processing, "width", 0) or 0),
                "height": int(getattr(processing, "height", 0) or 0),
                "steps": int(plan.steps),
                "guidance_scale": float(plan.guidance_scale),
                "sampler": getattr(processing, "sampler_name", None) or plan.sampler_name,
                "scheduler": plan.scheduler_name,
                "prompts": prompt_context.prompts,
                "negative_prompts": prompt_context.negative_prompts,
                "seeds": plan.seeds,
                "subseeds": plan.subseeds,
            },
        }
        effective_sampler_name = getattr(processing, "sampler_name", None) or plan.sampler_name
        if isinstance(effective_sampler_name, str) and effective_sampler_name.strip().lower() == "er sde":
            if plan.er_sde is None:
                payload["metadata"]["er_sde"] = {
                    "solver_type": "er_sde",
                    "max_stage": 3,
                    "eta": 1.0,
                    "s_noise": 1.0,
                }
            else:
                payload["metadata"]["er_sde"] = {
                    "solver_type": plan.er_sde.solver_type,
                    "max_stage": int(plan.er_sde.max_stage),
                    "eta": float(plan.er_sde.eta),
                    "s_noise": float(plan.er_sde.s_noise),
                }
        torch.save(payload, target)
        logger.info("[diagnostics] dumped latents to %s", target)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to dump latents to %s: %s", target, exc)


def execute_sampling(
    processing: Any,
    plan: SamplingPlan,
    payload: ConditioningPayload,
    prompt_context: PromptContext,
    prompt_loras: Sequence[Any],
    prompt_controls: Mapping[str, Any],
    *,
    rng: ImageRNG,
    noise: torch.Tensor | None = None,
    image_conditioning: torch.Tensor | None = None,
    init_latent: torch.Tensor | None = None,
    start_at_step: int | None = None,
    post_step_hook: Callable[[torch.Tensor, int, int], None] | None = None,
    post_sample_hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Execute the sampler using the provided configuration."""
    if noise is None:
        noise = rng.next()

    model = processing.sd_model
    if hasattr(model, "codex_objects_original") and model.codex_objects_original is not None:
        model.codex_objects = model.codex_objects_original.shallow_copy()

    run_before_sampling_hooks(processing, prompt_context, plan.seeds, plan.subseeds)

    merged = collect_lora_selections(prompt_loras)
    if merged:
        stats = apply_loras_to_engine(model, merged)
        logger.info("[native] Applied %d LoRA(s), %d params touched", stats.files, stats.params_touched)
    model.codex_objects = model.codex_objects_after_applying_lora.shallow_copy()

    if processing.scripts is not None:
        processing.scripts.process_before_every_sampling(
            processing,
            x=noise,
            noise=noise,
            c=payload.conditioning,
            uc=payload.unconditional,
        )

    if getattr(processing, "modified_noise", None) is not None:
        noise = processing.modified_noise
        processing.modified_noise = None

    preview_method = live_preview_method(default=LivePreviewMethod.FULL)
    debug_factors = debug_preview_factors_enabled()

    def _preview_cb(denoised_latent: torch.Tensor, step: int, total: int) -> None:
        # Skip preview decode on the final step; the engine will decode once
        # for the actual output, avoiding redundant VAE work at the tail.
        try:
            if total is not None and int(total) > 0 and int(step) >= int(total):
                if debug_factors:
                    maybe_log_preview_factors(processing, denoised_latent, step=int(step), total=int(total))
                return
        except Exception:
            # If step/total are malformed, fall back to best-effort preview.
            pass
        preview = decode_preview_image(processing, denoised_latent, method=preview_method)
        if preview is None:
            return
        backend_state.set_current_image(preview, sampling_step=int(step))
        if debug_factors:
            maybe_log_preview_factors(processing, denoised_latent, step=int(step), total=int(total or 0))

    if image_conditioning is None:
        image_conditioning = txt2img_conditioning(
            processing.sd_model,
            noise,
            processing.width,
            processing.height,
        )

    sampler_name = plan.sampler_name
    scheduler_name = plan.scheduler_name
    context = build_sampling_context(
        processing.sd_model,
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
        steps=int(plan.steps),
        noise_source=plan.noise_settings.source.value,
        eta_noise_seed_delta=plan.noise_settings.eta_noise_seed_delta,
        height=int(getattr(processing, "height", 0) or 0) or None,
        width=int(getattr(processing, "width", 0) or 0) or None,
        device=noise.device,
        dtype=noise.dtype,
    )

    samples = processing.sampler.sample(
        processing,
        noise,
        payload.conditioning,
        payload.unconditional,
        image_conditioning=image_conditioning,
        init_latent=init_latent,
        start_at_step=start_at_step,
        preview_callback=_preview_cb,
        post_step_hook=post_step_hook,
        post_sample_hook=post_sample_hook,
        context=context,
        er_sde_options=plan.er_sde,
    )

    samples = run_post_sample_hooks(processing, samples)
    _maybe_dump_latents(samples, processing, plan, prompt_context)
    devices.torch_gc()
    return samples
