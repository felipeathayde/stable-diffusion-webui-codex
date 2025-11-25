from __future__ import annotations

import logging
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
    return torch.cat([sigmas, torch.zeros(1, device=device, dtype=dtype)])


def build_sigma_schedule(
    scheduler_name: str,
    steps: int,
    *,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be >= 1")

    kind = SchedulerName.from_string(scheduler_name)
    if kind in (SchedulerName.AUTOMATIC, SchedulerName.SIMPLE):
        # Linear schedule from sigma_max -> sigma_min + terminal 0
        return torch.linspace(sigma_max, sigma_min, steps + 1, device=device, dtype=dtype)
    if kind in (SchedulerName.KARRAS, SchedulerName.EULER_DISCRETE):
        # Euler in diffusers commonly uses Karras sigmas; we purposefully
        # build the Karras schedule here. Integrator is selected elsewhere.
        return _karras_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    if kind is SchedulerName.EXPONENTIAL:
        return _exponential_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    # Exhaustive by construction, but keep explicit guard for clarity
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
) -> SamplingContext:
    sampler_kind = SamplerKind.from_string(sampler_name or "automatic")
    predictor = getattr(sd_model.codex_objects.unet, "model", None)
    if predictor is None or getattr(predictor, "predictor", None) is None:
        raise RuntimeError("sd_model does not expose a predictor for sigma bounds")

    pred = predictor.predictor
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

    dev = device or predictor.diffusion_model.load_device
    dt = dtype or getattr(predictor.diffusion_model, "dtype", torch.float32)

    sigmas = build_sigma_schedule(
        scheduler_name or "Automatic",
        steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=dev,
        dtype=dt,
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
def _exponential_schedule(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be >= 1")
    ramp = torch.linspace(0, 1, steps, device=device, dtype=dtype)
    sigmas = sigma_max * (sigma_min / sigma_max) ** ramp
    return torch.cat([sigmas, torch.zeros(1, device=device, dtype=dtype)])
