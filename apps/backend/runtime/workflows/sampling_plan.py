"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampling plan construction helpers for workflow orchestration.
Normalizes scheduler names/aliases, resolves noise settings, builds/overrides a `SamplingPlan`, and prepares the sampler + RNG.

Symbols (top-level; keep in sync; no ghosts):
- `_normalize_scheduler_name` (function): Normalize scheduler aliases and auto-tokens into a supported canonical scheduler name.
- `resolve_noise_settings` (function): Derive `NoiseSettings` for a run from processing overrides and env.
- `build_sampling_plan` (function): Build a `SamplingPlan` from processing state and explicit seeds/subseeds.
- `apply_sampling_overrides` (function): Apply prompt-derived overrides to a `SamplingPlan` (and reflect them into processing state).
- `ensure_sampler_and_rng` (function): Ensure `processing.sampler` and `processing.rng` exist for the current sampling plan.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Sequence

from apps.backend.core.rng import ImageRNG, NoiseSettings, NoiseSourceKind
from apps.backend.runtime.processing.datatypes import SamplingPlan
from apps.backend.runtime.sampling.catalog import (
    AUTO_TOKENS,
    SAMPLER_DEFAULT_SCHEDULER,
    SCHEDULER_ALIAS_TO_CANONICAL,
    SUPPORTED_SCHEDULERS,
)
from apps.backend.runtime.sampling.context import SchedulerName
from apps.backend.runtime.sampling.driver import CodexSampler

logger = logging.getLogger(__name__)


def _normalize_scheduler_name(sampler: str | None, scheduler: str | None) -> str:
    sampler_key = (sampler or "").strip().lower()
    raw = (scheduler or "").strip().lower()
    if raw in AUTO_TOKENS:
        raw = SAMPLER_DEFAULT_SCHEDULER.get(sampler_key, "automatic")
    canonical = SCHEDULER_ALIAS_TO_CANONICAL.get(raw, raw)
    if canonical not in SUPPORTED_SCHEDULERS:
        raise ValueError(f"Scheduler '{canonical}' is not supported")
    try:
        canonical_enum = SchedulerName.from_string(canonical)
    except ValueError as exc:
        raise ValueError(f"Unsupported scheduler '{scheduler}' for sampler '{sampler}'") from exc
    return canonical_enum.value


def resolve_noise_settings(processing: Any) -> NoiseSettings:
    """Inspect processing overrides/environment and return noise source settings."""
    source = None
    eta_delta = 0
    overrides = getattr(processing, "override_settings", {})
    if isinstance(overrides, dict):
        source = overrides.get("randn_source") or overrides.get("noise_source")
        eta_delta = overrides.get("eta_noise_seed_delta", eta_delta)
    metadata = getattr(processing, "metadata", {})
    if isinstance(metadata, dict):
        source = metadata.get("randn_source", source)
    if getattr(processing, "noise_source", None):
        source = processing.noise_source
    env_source = os.getenv("CODEX_NOISE_SOURCE")
    if source is None and env_source:
        source = env_source

    try:
        source_kind = NoiseSourceKind.from_string(source) if source else NoiseSourceKind.GPU
    except ValueError:
        source_kind = NoiseSourceKind.GPU

    delta = int(getattr(processing, "eta_noise_seed_delta", eta_delta) or eta_delta or 0)
    settings = NoiseSettings(source=source_kind, eta_noise_seed_delta=delta)
    processing.eta_noise_seed_delta = settings.eta_noise_seed_delta
    return settings


def build_sampling_plan(
    processing: Any,
    seeds: Sequence[int],
    subseeds: Sequence[int],
    subseed_strength: float,
    noise_settings: NoiseSettings | None = None,
) -> SamplingPlan:
    """Create a sampling plan for the generation run."""
    if noise_settings is None:
        noise_settings = resolve_noise_settings(processing)
    guidance = float(getattr(processing, "guidance_scale", 7.0) or 7.0)
    steps = int(getattr(processing, "steps", 20) or 20)
    sampler_name = getattr(processing, "sampler_name", None)
    scheduler_name = getattr(processing, "scheduler", None)
    try:
        normalized_scheduler = _normalize_scheduler_name(sampler_name, scheduler_name)
    except ValueError:
        logger.warning(
            "Invalid scheduler '%s' for sampler '%s'; falling back to automatic.",
            scheduler_name,
            sampler_name,
        )
        normalized_scheduler = SchedulerName.AUTOMATIC.value
    processing.scheduler = normalized_scheduler
    return SamplingPlan(
        sampler_name=sampler_name,
        scheduler_name=normalized_scheduler,
        steps=steps,
        guidance_scale=guidance,
        seeds=list(seeds),
        subseeds=list(subseeds),
        subseed_strength=float(subseed_strength),
        noise_settings=noise_settings,
    )


def apply_sampling_overrides(
    processing: Any,
    controls: Mapping[str, Any],
    plan: SamplingPlan,
) -> SamplingPlan:
    """Apply prompt-derived overrides to the sampling plan."""
    sampler_name = controls.get("sampler")
    if sampler_name:
        processing.sampler_name = str(sampler_name)
        plan.sampler_name = str(sampler_name)

    try:
        if "cfg" in controls:
            cfg = float(controls["cfg"])
            processing.guidance_scale = cfg
            processing.cfg_scale = cfg
            plan.guidance_scale = cfg
        if "steps" in controls:
            steps = int(float(controls["steps"]))
            processing.steps = steps
            plan.steps = steps
        if "seed" in controls:
            seed = int(float(controls["seed"]))
            plan.seeds = [seed]
            processing.seeds = [seed]
    except Exception:
        logger.debug("Failed to apply sampling overrides", exc_info=True)
    return plan


def ensure_sampler_and_rng(
    processing: Any,
    plan: SamplingPlan,
    *,
    latent_channels: int | None = None,
) -> ImageRNG:
    """Ensure processing has a sampler + RNG configured for the current plan."""
    algo = plan.sampler_name or getattr(processing, "sampler_name", None)
    processing.sampler = CodexSampler(processing.sd_model, algorithm=algo)
    if latent_channels is None:
        latent_channels = getattr(
            processing.sd_model.codex_objects_after_applying_lora.vae,
            "latent_channels",
            4,
        )
    shape = (
        latent_channels,
        processing.height // 8,
        processing.width // 8,
    )
    rng = ImageRNG(
        shape,
        plan.seeds,
        subseeds=plan.subseeds,
        subseed_strength=plan.subseed_strength,
        seed_resize_from_h=getattr(processing, "seed_resize_from_h", 0),
        seed_resize_from_w=getattr(processing, "seed_resize_from_w", 0),
        settings=plan.noise_settings,
    )
    processing.rng = rng
    return rng

