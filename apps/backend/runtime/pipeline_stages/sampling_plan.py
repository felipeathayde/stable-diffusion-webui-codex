"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampling plan construction helpers for pipeline orchestration.
Validates sampler/scheduler selection, resolves noise settings, builds/overrides a `SamplingPlan`, and prepares the sampler + RNG.

Symbols (top-level; keep in sync; no ghosts):
- `_normalize_scheduler_name` (function): Validate a scheduler name for the given sampler.
- `resolve_sampler_scheduler_override` (function): Resolve sampler/scheduler for a derived plan (e.g., hires pass) with override semantics.
- `resolve_noise_settings` (function): Derive `NoiseSettings` for a run from processing overrides and env.
- `resolve_er_sde_options` (function): Build normalized typed ER-SDE options from processing overrides.
- `build_sampling_plan` (function): Build a `SamplingPlan` from processing state and explicit seeds/subseeds.
- `apply_sampling_overrides` (function): Apply prompt-derived overrides to a `SamplingPlan` (and reflect them into processing state).
- `ensure_sampler_and_rng` (function): Ensure `processing.sampler` and `processing.rng` exist for the current sampling plan.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from apps.backend.core.rng import ImageRNG, NoiseSettings, NoiseSourceKind
from apps.backend.runtime.processing.datatypes import ErSdeOptions, SamplingPlan
from apps.backend.runtime.sampling import SUPPORTED_SCHEDULERS
from apps.backend.runtime.sampling.context import SchedulerName
from apps.backend.runtime.sampling.driver import CodexSampler
from apps.backend.runtime.sampling.registry import get_sampler_spec

logger = logging.getLogger(__name__)


def _normalize_scheduler_name(sampler: str, scheduler: str) -> str:
    if scheduler not in SUPPORTED_SCHEDULERS:
        raise ValueError(f"Scheduler '{scheduler}' is not supported")
    try:
        canonical_enum = SchedulerName.from_string(scheduler)
    except ValueError as exc:
        raise ValueError(f"Unsupported scheduler '{scheduler}'") from exc
    spec = get_sampler_spec(sampler)
    if not spec.is_supported_scheduler(canonical_enum.value):
        raise ValueError(f"Scheduler '{scheduler}' is not supported for sampler '{sampler}'")
    return canonical_enum.value


def resolve_sampler_scheduler_override(
    *,
    base_sampler: str,
    base_scheduler: str,
    sampler_override: str | None,
    scheduler_override: str | None,
) -> tuple[str, str]:
    """Resolve sampler/scheduler selection for a derived sampling plan (e.g., hires pass).

    Semantics:
    - If `sampler_override` is set, it becomes the sampler for the derived plan.
      - If `scheduler_override` is NOT set, the scheduler defaults to the sampler's default scheduler.
    - If only `scheduler_override` is set, it is validated against the base sampler.
    - If neither override is set, base sampler/scheduler are kept.
    """

    base_sampler_value = str(base_sampler or "").strip()
    base_scheduler_value = str(base_scheduler or "").strip()
    if not base_sampler_value:
        raise ValueError("base_sampler must be a non-empty sampler name")
    if not base_scheduler_value:
        raise ValueError("base_scheduler must be a non-empty scheduler name")

    def _normalize_override(value: str | None, *, kind: str) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{kind}_override must be a string when provided")
        normalized = value.strip()
        if normalized == "":
            return ""
        lowered = normalized.lower()
        if kind == "sampler" and lowered in {"use same sampler", "use same"}:
            return ""
        if kind == "scheduler" and lowered in {"use same scheduler", "use same"}:
            return ""
        return normalized

    sampler_override_value = _normalize_override(sampler_override, kind="sampler")
    scheduler_override_value = _normalize_override(scheduler_override, kind="scheduler")

    sampler_name = sampler_override_value or base_sampler_value
    if scheduler_override_value:
        scheduler_name = scheduler_override_value
    elif sampler_override_value:
        scheduler_name = get_sampler_spec(sampler_name).default_scheduler
    else:
        scheduler_name = base_scheduler_value

    normalized_scheduler = _normalize_scheduler_name(sampler_name, scheduler_name)
    return sampler_name, normalized_scheduler


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

    try:
        source_kind = NoiseSourceKind.from_string(source) if source else NoiseSourceKind.GPU
    except ValueError:
        source_kind = NoiseSourceKind.GPU

    delta = int(getattr(processing, "eta_noise_seed_delta", eta_delta) or eta_delta or 0)
    settings = NoiseSettings(source=source_kind, eta_noise_seed_delta=delta)
    processing.eta_noise_seed_delta = settings.eta_noise_seed_delta
    return settings


def resolve_er_sde_options(processing: Any) -> ErSdeOptions | None:
    """Resolve ER-SDE options from processing overrides with strict validation."""
    overrides = getattr(processing, "override_settings", {})
    if not isinstance(overrides, dict):
        return None
    if "er_sde" not in overrides:
        return None
    normalized = CodexSampler._resolve_er_sde_runtime_params(overrides.get("er_sde"))
    return ErSdeOptions(
        solver_type=str(normalized["solver_type"]),
        max_stage=int(normalized["max_stage"]),
        eta=float(normalized["eta"]),
        s_noise=float(normalized["s_noise"]),
    )


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
    if not isinstance(sampler_name, str) or not sampler_name:
        raise ValueError("processing.sampler_name must be set to a non-empty sampler name")
    if not isinstance(scheduler_name, str) or not scheduler_name:
        raise ValueError("processing.scheduler must be set to a non-empty scheduler name")
    normalized_scheduler = _normalize_scheduler_name(sampler_name, scheduler_name)
    processing.scheduler = normalized_scheduler
    hr_sampler_name = getattr(processing, "hr_sampler_name", None)
    er_sde_in_use = sampler_name.strip().lower() == "er sde"
    if isinstance(hr_sampler_name, str) and hr_sampler_name.strip().lower() == "er sde":
        er_sde_in_use = True
    return SamplingPlan(
        sampler_name=sampler_name,
        scheduler_name=normalized_scheduler,
        steps=steps,
        guidance_scale=guidance,
        seeds=list(seeds),
        subseeds=list(subseeds),
        subseed_strength=float(subseed_strength),
        noise_settings=noise_settings,
        er_sde=resolve_er_sde_options(processing) if er_sde_in_use else None,
    )


def apply_sampling_overrides(
    processing: Any,
    controls: Mapping[str, Any],
    plan: SamplingPlan,
) -> SamplingPlan:
    """Apply prompt-derived overrides to the sampling plan."""
    sampler_raw = controls.get("sampler")
    scheduler_raw = controls.get("scheduler")
    sampler_changed = False
    scheduler_changed = False
    if sampler_raw:
        processing.sampler_name = str(sampler_raw)
        plan.sampler_name = str(sampler_raw)
        sampler_changed = True
    if scheduler_raw:
        processing.scheduler = str(scheduler_raw)
        plan.scheduler_name = str(scheduler_raw)
        scheduler_changed = True
    if sampler_changed and not scheduler_changed:
        spec = get_sampler_spec(str(plan.sampler_name))
        processing.scheduler = spec.default_scheduler
        plan.scheduler_name = spec.default_scheduler

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
    hr_sampler_name = getattr(processing, "hr_sampler_name", None)
    er_sde_in_use = (
        isinstance(plan.sampler_name, str) and plan.sampler_name.strip().lower() == "er sde"
    ) or (
        isinstance(hr_sampler_name, str) and hr_sampler_name.strip().lower() == "er sde"
    )
    plan.er_sde = resolve_er_sde_options(processing) if er_sde_in_use else None
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
