"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampling context + sigma schedule construction utilities for diffusion samplers.
Defines the canonical scheduler names and builds sigma schedules (Karras, exponential, DDIM, align-your-steps, etc.), then packages
per-run sampling state (sampler kind, noise settings, scheduler config) into a `SamplingContext`.

Symbols (top-level; keep in sync; no ghosts):
- `SchedulerName` (enum): Canonical scheduler names for sigma schedule construction (alias-aware, no silent fallback).
- `_append_zero` (function): Appends a terminal sigma=0 to a sigma schedule tensor.
- `_karras_schedule` (function): Builds a Karras sigma schedule.
- `_polyexponential_schedule` (function): Builds a polyexponential sigma schedule.
- `_exponential_schedule` (function): Builds an exponential sigma schedule.
- `_uniform_schedule_from_predictor` (function): Builds a uniform schedule from a predictor function.
- `_simple_schedule_from_predictor` (function): Builds a simple schedule from a predictor function.
- `_sgm_uniform_schedule` (function): Builds an SGM-uniform schedule (SGM-style).
- `_ddim_uniform_schedule` (function): Builds a DDIM-uniform schedule.
- `_normal_schedule` (function): Builds a normal/beta-derived schedule (optionally SGM variant).
- `_beta_schedule` (function): Builds a beta schedule.
- `_linear_quadratic_schedule` (function): Builds a linear-quadratic schedule.
- `_kl_optimal_schedule` (function): Builds a KL-optimal schedule.
- `_align_your_steps_schedule` (function): Builds the “align your steps” schedule variants (SDXL aware).
- `_turbo_schedule` (function): Builds a turbo schedule.
- `build_sigma_schedule` (function): Main scheduler entrypoint; selects the schedule builder and returns the sigma tensor.
- `_env_flag` (function): Reads boolean toggles from env for debug/feature flags affecting sampling context.
- `SamplingContext` (dataclass): Bundles sampling configuration/state for one run (sampler kind, scheduler, noise settings, etc.).
- `build_sampling_context` (function): Builds a `SamplingContext` from inputs (engine/runtime settings + request payload).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from apps.backend.core.rng import NoiseSettings, NoiseSourceKind
from apps.backend.engines.util.schedulers import SamplerKind
from apps.backend.runtime.sampling.catalog import SCHEDULER_ALIAS_TO_CANONICAL


_LOGGER = logging.getLogger(__name__ + ".context")


class SchedulerName(str, Enum):
    """Canonical scheduler names for sigma schedule construction.

    Notes
    - This controls ONLY the sigma schedule construction, not the integrator
      (which is selected via `SamplerKind`).
    - We accept a limited set of aliases coming from UI/API or diffusers names,
      but do not silently fallback: unknown values raise with a clear message.
    """

    AUTOMATIC = "automatic"
    SIMPLE = "simple"
    KARRAS = "karras"
    EULER_DISCRETE = "euler_discrete"
    EXPONENTIAL = "exponential"
    POLYEXPONENTIAL = "polyexponential"
    UNIFORM = "uniform"
    SGM_UNIFORM = "sgm_uniform"
    DDIM = "ddim"
    DDIM_UNIFORM = "ddim_uniform"
    NORMAL = "normal"
    BETA = "beta"
    LINEAR_QUADRATIC = "linear_quadratic"
    KL_OPTIMAL = "kl_optimal"
    TURBO = "turbo"
    ALIGN_YOUR_STEPS = "align_your_steps"
    ALIGN_YOUR_STEPS_GITS = "align_your_steps_gits"
    ALIGN_YOUR_STEPS_11 = "align_your_steps_11"
    ALIGN_YOUR_STEPS_32 = "align_your_steps_32"

    @staticmethod
    def from_string(name: str | None) -> "SchedulerName":
        key = (name or "automatic").strip().lower()
        canonical = SCHEDULER_ALIAS_TO_CANONICAL.get(key, key)
        try:
            return SchedulerName(canonical)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported scheduler '{name}'. Supported: {[m.value for m in SchedulerName]}"
            ) from exc
def _append_zero(sigmas: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    terminal = torch.zeros(1, device=device, dtype=dtype)
    return torch.cat([sigmas.to(device=device, dtype=dtype), terminal])


def _karras_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rho: float = 7.0,
) -> torch.Tensor:
    ramp = torch.linspace(0, 1, steps, device=device, dtype=dtype)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + (min_inv - max_inv) * ramp) ** rho
    return _append_zero(sigmas, device=device, dtype=dtype)


def _polyexponential_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rho: float = 1.0,
) -> torch.Tensor:
    ramp = torch.linspace(1, 0, steps, device=device, dtype=dtype) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return _append_zero(sigmas, device=device, dtype=dtype)


def _exponential_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ramp = torch.linspace(math.log(sigma_max), math.log(sigma_min), steps, device=device, dtype=dtype)
    sigmas = ramp.exp()
    return _append_zero(sigmas, device=device, dtype=dtype)


def _uniform_schedule_from_predictor(steps: int, predictor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigmas = getattr(predictor, "sigmas", None)
    if sigmas is None:
        raise RuntimeError("predictor does not expose 'sigmas' needed for uniform schedule")
    sigmas = torch.as_tensor(sigmas, device=device, dtype=dtype)
    # Evenly sample sigmas from the predictor ladder (skip sigma=inf at index 0)
    stride = max(int(math.floor(len(sigmas) / max(steps, 1))), 1)
    ladder = sigmas[1::stride][:steps]
    if ladder.numel() < steps:
        ladder = torch.nn.functional.interpolate(
            ladder.view(1, 1, -1), size=steps, mode="linear", align_corners=False
        ).view(-1)
    return _append_zero(ladder, device=device, dtype=dtype)


def _simple_schedule_from_predictor(steps: int, predictor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Predictor-ladder 'simple' schedule built via predictor sigma/timestep.

    Mirrors the reference schedule-linker `get_sigmas(n)` behavior:
    - Construct a linear ladder of timesteps from high→low over the predictor domain.
    - Map timesteps back to sigmas via `predictor.sigma(t)`.
    - Append a terminal 0 as the last sigma.
    """
    # Special case: FlowMatchEulerPrediction (Z Image Turbo) expects the diffusers
    # `ZImagePipeline` schedule shape:
    # - The pipeline sets `scheduler.sigma_min = 0.0` and calls `set_timesteps(N)`.
    # - This yields base sigmas linearly spaced from 1.0 -> 0.0 (inclusive),
    #   then applies the `shift` transform once (and appends a terminal 0).
    # - Because 0.0 is already present as the final inference sigma, the schedule
    #   ends with a double-zero tail (dt=0 for the last step). This matches the
    #   upstream recommendation `num_inference_steps=9` to get 8 effective steps.
    mu = getattr(predictor, "mu", None)
    pseudo = getattr(predictor, "pseudo_timestep_range", None)
    if mu is not None and int(pseudo or 0) == 1000:
        try:
            mu_value = float(mu)
        except Exception:  # noqa: BLE001 - defensive
            mu_value = 0.0
        if mu_value > 0.0:
            base = torch.linspace(1.0, 0.0, int(steps), device=device, dtype=dtype)
            if mu_value == 1.0:
                shifted = base
            else:
                shifted = mu_value * base / (1.0 + (mu_value - 1.0) * base)
            return _append_zero(shifted, device=device, dtype=dtype)

    base_sigmas = getattr(predictor, "sigmas", None)
    if base_sigmas is None:
        raise RuntimeError("predictor does not expose 'sigmas' needed for simple scheduler")
    total = int(len(base_sigmas))
    if total == 0:
        raise RuntimeError("predictor.sigmas is empty; cannot build simple sigma schedule")

    t_max = total - 1
    t = torch.linspace(float(t_max), 0.0, int(steps), device=device, dtype=dtype)
    ladder = predictor.sigma(t)
    return _append_zero(ladder, device=device, dtype=dtype)


def _sgm_uniform_schedule(steps: int, predictor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    start = predictor.sigma_to_t(torch.as_tensor(float(predictor.sigma_max), device=device, dtype=dtype))
    end = predictor.sigma_to_t(torch.as_tensor(float(predictor.sigma_min), device=device, dtype=dtype))
    timesteps = torch.linspace(start, end, steps + 1, device=device, dtype=dtype)[:-1]
    sigmas = predictor.t_to_sigma(timesteps)
    return _append_zero(sigmas, device=device, dtype=dtype)


def _ddim_uniform_schedule(steps: int, predictor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigmas = torch.as_tensor(getattr(predictor, "sigmas", []), device=device, dtype=dtype)
    if sigmas.numel() == 0:
        raise RuntimeError("predictor is missing sigmas ladder required for DDIM uniform schedule")
    stride = max(len(sigmas) // max(steps, 1), 1)
    ladder = sigmas[1::stride]
    ladder = ladder.flip(0)[:steps]
    return _append_zero(ladder, device=device, dtype=dtype)


def _normal_schedule(steps: int, predictor, *, device: torch.device, dtype: torch.dtype, sgm: bool = False) -> torch.Tensor:
    start = predictor.timestep(torch.as_tensor(float(predictor.sigma_max), device=device, dtype=dtype))
    end = predictor.timestep(torch.as_tensor(float(predictor.sigma_min), device=device, dtype=dtype))
    if sgm:
        timesteps = torch.linspace(start, end, steps + 1, device=device, dtype=dtype)[:-1]
    else:
        timesteps = torch.linspace(start, end, steps, device=device, dtype=dtype)
    sigmas = predictor.sigma(timesteps)
    return _append_zero(sigmas, device=device, dtype=dtype)


def _beta_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    alpha: float = 0.6,
    beta: float = 0.6,
) -> torch.Tensor:
    # Using torch.distributions avoids scipy dependency
    dist = torch.distributions.Beta(torch.as_tensor(alpha, device=device), torch.as_tensor(beta, device=device))
    lin = torch.linspace(0, 1, steps, device=device, dtype=dtype)
    timesteps = dist.icdf(lin.clamp(1e-6, 1 - 1e-6))
    sigmas = sigma_min + (sigma_max - sigma_min) * (1 - timesteps)
    return _append_zero(sigmas, device=device, dtype=dtype)


def _linear_quadratic_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    threshold_noise: float = 0.025,
    linear_steps: int | None = None,
) -> torch.Tensor:
    if steps == 1:
        sigmas = torch.tensor([sigma_max], device=device, dtype=dtype)
        return _append_zero(sigmas, device=device, dtype=dtype)
    if linear_steps is None:
        linear_steps = steps // 2
    linear = torch.linspace(0, threshold_noise, linear_steps, device=device, dtype=dtype)
    threshold_noise_step_diff = linear_steps - threshold_noise * steps
    quadratic_steps = steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / max(quadratic_steps ** 2, 1e-6)
    linear_coef = threshold_noise / max(linear_steps, 1e-6) - 2 * threshold_noise_step_diff / max(quadratic_steps ** 2, 1e-6)
    const = quadratic_coef * (linear_steps ** 2)
    quadratic = quadratic_coef * torch.arange(linear_steps, steps, device=device, dtype=dtype) ** 2 + linear_coef * torch.arange(linear_steps, steps, device=device, dtype=dtype) + const
    sigma_schedule = torch.cat([linear, quadratic, torch.tensor([threshold_noise], device=device, dtype=dtype)])
    sigma_schedule = (1.0 - sigma_schedule).clamp(min=0.0)
    sigmas = sigma_schedule * sigma_max
    return _append_zero(sigmas[:steps], device=device, dtype=dtype)


def _kl_optimal_schedule(steps: int, sigma_min: float, sigma_max: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(steps, device=device, dtype=dtype) / max(float(steps - 1), 1.0)
    sigmas = torch.tan(idx * math.atan(sigma_min) + (1 - idx) * math.atan(sigma_max))
    return _append_zero(sigmas, device=device, dtype=dtype)


def _align_your_steps_schedule(kind: SchedulerName, steps: int, *, is_sdxl: bool, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Tables mirrored from the reference sd_schedulers tables; trailing zero will be appended separately
    if kind is SchedulerName.ALIGN_YOUR_STEPS:
        sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029] if is_sdxl else [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]
    elif kind is SchedulerName.ALIGN_YOUR_STEPS_GITS:
        sigmas = [14.615, 4.734, 2.567, 1.529, 0.987, 0.652, 0.418, 0.268, 0.179, 0.127, 0.029] if is_sdxl else [14.615, 4.617, 2.507, 1.236, 0.702, 0.402, 0.240, 0.156, 0.104, 0.094, 0.029]
    elif kind is SchedulerName.ALIGN_YOUR_STEPS_11:
        sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029] if is_sdxl else [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]
    else:  # ALIGN_YOUR_STEPS_32
        if is_sdxl:
            sigmas = [14.61500000000000000, 11.14916180000000000, 8.505221270000000000, 6.488271510000000000, 5.437074020000000000, 4.603986190000000000, 3.898547040000000000, 3.274074570000000000, 2.743965270000000000, 2.299686590000000000, 1.954485140000000000, 1.671087150000000000, 1.428781520000000000, 1.231810090000000000, 1.067896490000000000, 0.925794430000000000, 0.802908860000000000, 0.696601210000000000, 0.604369030000000000, 0.528525520000000000, 0.467733440000000000, 0.413933790000000000, 0.362581860000000000, 0.310085170000000000, 0.265189250000000000, 0.223264610000000000, 0.176538770000000000, 0.139591920000000000, 0.105873810000000000, 0.055193690000000000, 0.028773340000000000, 0.015000000000000000]
        else:
            sigmas = [14.61500000000000000, 11.23951352000000000, 8.643630810000000000, 6.647294240000000000, 5.572508620000000000, 4.716485460000000000, 3.991960650000000000, 3.519560900000000000, 3.134904660000000000, 2.792287880000000000, 2.487736280000000000, 2.216638650000000000, 1.975083510000000000, 1.779317200000000000, 1.614753350000000000, 1.465409530000000000, 1.314849000000000000, 1.166424970000000000, 1.034755470000000000, 0.915737440000000000, 0.807481690000000000, 0.712023610000000000, 0.621739000000000000, 0.530652020000000000, 0.452909600000000000, 0.374914550000000000, 0.274618190000000000, 0.201152900000000000, 0.141058730000000000, 0.066828810000000000, 0.031661210000000000, 0.015000000000000000]
    sigmas = torch.as_tensor(sigmas, device=device, dtype=dtype)
    if sigmas.numel() != steps:
        sigmas = torch.nn.functional.interpolate(sigmas.view(1, 1, -1), size=steps, mode="linear", align_corners=False).view(-1)
    return _append_zero(sigmas, device=device, dtype=dtype)


def _turbo_schedule(steps: int, predictor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    timesteps = torch.flip(torch.arange(1, steps + 1, device=device, dtype=dtype) * float(1000.0 / steps) - 1, (0,)).round().clamp(0, 999)
    sigmas = predictor.sigma(timesteps)
    return _append_zero(sigmas, device=device, dtype=dtype)


def build_sigma_schedule(
    scheduler_name: str,
    steps: int,
    *,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    dtype: torch.dtype,
    predictor: Optional[object] = None,
    is_sdxl: bool = False,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be >= 1")

    kind = SchedulerName.from_string(scheduler_name)

    if kind is SchedulerName.AUTOMATIC:
        # Linear sigma ramp with a terminal 0 (matches sampler expectation).
        sigmas = torch.linspace(sigma_max, sigma_min, steps, device=device, dtype=dtype)
        return _append_zero(sigmas, device=device, dtype=dtype)
    if kind is SchedulerName.SIMPLE:
        if predictor is None:
            raise RuntimeError("predictor required for simple scheduler")
        return _simple_schedule_from_predictor(steps, predictor, device=device, dtype=dtype)
    if kind in (SchedulerName.KARRAS, SchedulerName.EULER_DISCRETE):
        return _karras_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    if kind is SchedulerName.EXPONENTIAL:
        return _exponential_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    if kind is SchedulerName.POLYEXPONENTIAL:
        return _polyexponential_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    if kind is SchedulerName.UNIFORM:
        if predictor is None:
            raise RuntimeError("predictor required for uniform scheduler")
        return _uniform_schedule_from_predictor(steps, predictor, device=device, dtype=dtype)
    if kind is SchedulerName.SGM_UNIFORM:
        if predictor is None:
            raise RuntimeError("predictor required for sgm_uniform scheduler")
        return _sgm_uniform_schedule(steps, predictor, device=device, dtype=dtype)
    if kind is SchedulerName.DDIM:
        if predictor is None:
            raise RuntimeError("predictor required for ddim scheduler")
        sigs = _ddim_uniform_schedule(steps, predictor, device=device, dtype=dtype)
        return sigs
    if kind is SchedulerName.DDIM_UNIFORM:
        if predictor is None:
            raise RuntimeError("predictor required for ddim_uniform scheduler")
        return _ddim_uniform_schedule(steps, predictor, device=device, dtype=dtype)
    if kind is SchedulerName.NORMAL:
        if predictor is None:
            raise RuntimeError("predictor required for normal scheduler")
        return _normal_schedule(steps, predictor, device=device, dtype=dtype, sgm=False)
    if kind is SchedulerName.BETA:
        return _beta_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    if kind is SchedulerName.LINEAR_QUADRATIC:
        return _linear_quadratic_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    if kind is SchedulerName.KL_OPTIMAL:
        return _kl_optimal_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    if kind in {
        SchedulerName.ALIGN_YOUR_STEPS,
        SchedulerName.ALIGN_YOUR_STEPS_GITS,
        SchedulerName.ALIGN_YOUR_STEPS_11,
        SchedulerName.ALIGN_YOUR_STEPS_32,
    }:
        return _align_your_steps_schedule(kind, steps, is_sdxl=is_sdxl, device=device, dtype=dtype)
    if kind is SchedulerName.TURBO:
        if predictor is None:
            raise RuntimeError("predictor required for turbo scheduler")
        return _turbo_schedule(steps, predictor, device=device, dtype=dtype)

    raise ValueError(f"Unsupported scheduler '{scheduler_name}' after normalization")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class SamplingContext:
    sampler_kind: SamplerKind
    scheduler_name: str
    sigmas: torch.Tensor
    steps: int
    noise_settings: NoiseSettings
    preview_interval: int = 0
    enable_progress: bool = False
    prediction_type: str | None = None
    sigma_min: float | None = None
    sigma_max: float | None = None
    sigma_data: float | None = None

    @property
    def device(self) -> torch.device:
        return self.sigmas.device


def build_sampling_context(
    sd_model,
    *,
    sampler_name: str | None,
    scheduler_name: str | None,
    steps: int,
    noise_source: str | None,
    eta_noise_seed_delta: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    predictor: Optional[object] = None,
    is_sdxl: bool = False,
) -> SamplingContext:
    sampler_kind = SamplerKind.from_string(sampler_name or "automatic")
    predictor_container = predictor or getattr(sd_model.codex_objects.denoiser, "model", None)
    if predictor_container is None or getattr(predictor_container, "predictor", None) is None:
        raise RuntimeError("sd_model does not expose a predictor for sigma bounds")

    pred = predictor_container.predictor
    def _as_float(value: object | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value.item()) if hasattr(value, "item") else float(value)  # type: ignore[arg-type]
        except Exception:
            return None

    sigma_min = _as_float(getattr(pred, "sigma_min", None))
    sigma_max = _as_float(getattr(pred, "sigma_max", None))
    if sigma_min is None or sigma_max is None:
        raise RuntimeError("predictor is missing sigma_min/sigma_max required for sampling")
    if sigma_max < sigma_min:
        sigma_min, sigma_max = sigma_max, sigma_min
    prediction_type = getattr(pred, "prediction_type", None)
    if isinstance(prediction_type, str):
        prediction_type = prediction_type.lower()
    sigma_data = _as_float(getattr(pred, "sigma_data", None))

    noise_settings = NoiseSettings(
        source=NoiseSourceKind.from_string(noise_source) if noise_source else NoiseSourceKind.GPU,
        eta_noise_seed_delta=int(eta_noise_seed_delta or 0),
    )

    dev = device or predictor_container.diffusion_model.load_device
    dt = dtype or getattr(predictor_container.diffusion_model, "dtype", torch.float32)

    sigmas = build_sigma_schedule(
        scheduler_name or "Automatic",
        steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=dev,
        dtype=dt,
        predictor=pred,
        is_sdxl=is_sdxl or bool(getattr(sd_model, "is_sdxl", False)),
    )

    context = SamplingContext(
        sampler_kind=sampler_kind,
        scheduler_name=scheduler_name or "Automatic",
        sigmas=sigmas,
        steps=steps,
        noise_settings=noise_settings,
        preview_interval=int(os.getenv("CODEX_PREVIEW_INTERVAL", "0") or 0),
        enable_progress=_env_flag("CODEX_PROGRESS_BAR", default=False),
        prediction_type=prediction_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
    )

    _LOGGER.debug(
        "sampling-context sampler=%s scheduler=%s steps=%d source=%s eta_delta=%d prediction=%s sigma_min=%.6g sigma_max=%.6g",
        context.sampler_kind.value,
        context.scheduler_name,
        context.steps,
        context.noise_settings.source.value,
        context.noise_settings.eta_noise_seed_delta,
        context.prediction_type,
        float(context.sigma_min) if context.sigma_min is not None else float("nan"),
        float(context.sigma_max) if context.sigma_max is not None else float("nan"),
    )
    return context


__all__ = [
    "SamplingContext",
    "build_sampling_context",
    "build_sigma_schedule",
    "SchedulerName",
]
