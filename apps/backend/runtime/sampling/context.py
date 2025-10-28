from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

from apps.backend.core.rng import NoiseSettings, NoiseSourceKind
from apps.backend.engines.util.schedulers import SamplerKind


_LOGGER = logging.getLogger(__name__ + ".context")


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

    name = (scheduler_name or "Automatic").strip().lower()
    if name in {"automatic", "simple"}:
        return torch.linspace(sigma_max, sigma_min, steps + 1, device=device, dtype=dtype)
    if name == "karras":
        return _karras_schedule(steps, sigma_min, sigma_max, device=device, dtype=dtype)
    raise ValueError(f"Unsupported scheduler '{scheduler_name}'")


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
    predictor = getattr(sd_model.forge_objects.unet, "model", None)
    if predictor is None or getattr(predictor, "predictor", None) is None:
        raise RuntimeError("sd_model does not expose a predictor for sigma bounds")

    pred = predictor.predictor
    sigma_min = float(pred.sigma_min.item() if hasattr(pred.sigma_min, "item") else pred.sigma_min)
    sigma_max = float(pred.sigma_max.item() if hasattr(pred.sigma_max, "item") else pred.sigma_max)
    if sigma_max < sigma_min:
        sigma_min, sigma_max = sigma_max, sigma_min

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
    )

    _LOGGER.debug(
        "sampling-context sampler=%s scheduler=%s steps=%d source=%s eta_delta=%d",
        context.sampler_kind.value,
        context.scheduler_name,
        context.steps,
        context.noise_settings.source.value,
        context.noise_settings.eta_noise_seed_delta,
    )
    return context


__all__ = [
    "SamplingContext",
    "build_sampling_context",
    "build_sigma_schedule",
]

