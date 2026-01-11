"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Live preview helpers (decode strategies + preview-factor fitting/logging).
Provides live preview decoding (full VAE vs cheap approximation) and an optional least-squares fitting tool to derive latent→RGB factors for
debugging preview quality.

Symbols (top-level; keep in sync; no ghosts):
- `debug_preview_factors_enabled` (function): Indicates whether preview factor fitting logs are enabled.
- `debug_preview_factors_sample_limit` (function): Returns the pixel sample cap used for factor fitting.
- `LivePreviewMethod` (enum): Preview decode strategy (`Full` VAE vs `Approx cheap`).
- `live_preview_method_from_env` (function): Reads `CODEX_LIVE_PREVIEW_METHOD` into a `LivePreviewMethod`.
- `live_preview_method_to_env` (function): Converts a `LivePreviewMethod` into an env-friendly string.
- `_tensor_to_pil_rgb` (function): Converts a tensor image into a PIL RGB image.
- `decode_preview_image` (function): Decodes a denoised latent into a preview image using the selected method.
- `PreviewFactorsFit` (dataclass): Fit result container for latent→RGB factors and bias (with MSE and VAE metadata).
- `fit_preview_factors` (function): Fits latent→RGB factors via least squares against a decoded VAE image (debug tool).
- `maybe_log_preview_factors` (function): Logs preview-factor fits once per job when enabled.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from apps.backend.core.state import state as backend_state
from apps.backend.infra.config.env_flags import env_flag, env_int

logger = logging.getLogger(__name__)

_LATENT_RGB_FACTORS_SD15: tuple[tuple[float, float, float], ...] = (
    (0.3512, 0.2297, 0.3227),
    (0.3250, 0.4974, 0.2350),
    (-0.2829, 0.1762, 0.2721),
    (-0.2120, -0.2616, -0.7177),
)

_LATENT_RGB_FACTORS_SDXL: tuple[tuple[float, float, float], ...] = (
    (0.3920, 0.4054, 0.4549),
    (-0.2634, -0.0196, 0.0653),
    (0.0568, 0.1687, -0.0755),
    (-0.3112, -0.2359, -0.2076),
)

_DEBUG_PREVIEW_FACTORS_LAST_JOB_TS: str | None = None


def debug_preview_factors_enabled() -> bool:
    return env_flag("CODEX_DEBUG_PREVIEW_FACTORS", default=False)


def debug_preview_factors_sample_limit() -> int:
    return env_int("CODEX_DEBUG_PREVIEW_FACTORS_SAMPLES", default=4096, min_value=256)


class LivePreviewMethod(str, Enum):
    FULL = "Full"
    APPROX_CHEAP = "Approx cheap"

    @staticmethod
    def from_string(value: str | None, *, default: "LivePreviewMethod" = FULL) -> "LivePreviewMethod":
        key = (value or "").strip().lower()
        if key in {"full", "vae", ""}:
            return LivePreviewMethod.FULL
        if key in {"approx cheap", "approx_cheap", "approx-cheap", "cheap"}:
            return LivePreviewMethod.APPROX_CHEAP
        return default


def live_preview_method_from_env(*, default: LivePreviewMethod = LivePreviewMethod.FULL) -> LivePreviewMethod:
    return LivePreviewMethod.from_string(os.getenv("CODEX_LIVE_PREVIEW_METHOD"), default=default)


def live_preview_method_to_env(method: LivePreviewMethod) -> str:
    return method.value


def _tensor_to_pil_rgb(tensor: Any) -> Any:
    import numpy as np
    from PIL import Image

    arr = tensor.detach().float().cpu().clamp(-1, 1)
    arr = ((arr + 1.0) * 0.5).mul(255.0).byte().movedim(0, -1).numpy()
    return Image.fromarray(np.asarray(arr), mode="RGB")


def decode_preview_image(processing: Any, denoised_latent: Any, *, method: LivePreviewMethod) -> Any | None:
    import torch
    import torch.nn.functional as F

    from apps.backend.runtime.processing.conditioners import decode_latent_batch

    if not isinstance(denoised_latent, torch.Tensor):
        return None
    if denoised_latent.ndim != 4:
        return None

    resolved = method
    if resolved == LivePreviewMethod.APPROX_CHEAP:
        if denoised_latent.shape[1] != 4:
            logger.warning(
                "Live preview method '%s' requires 4-channel latents; falling back to VAE decode.",
                method.value,
            )
            resolved = LivePreviewMethod.FULL
        else:
            factors = _LATENT_RGB_FACTORS_SDXL if bool(getattr(processing.sd_model, "is_sdxl", False)) else _LATENT_RGB_FACTORS_SD15
            mat = torch.tensor(factors, device=denoised_latent.device, dtype=denoised_latent.dtype)
            rgb_small = torch.einsum("blhw,lr->brhw", denoised_latent, mat)
            rgb = F.interpolate(rgb_small, scale_factor=8, mode="bilinear", align_corners=False)
            return _tensor_to_pil_rgb(rgb[0])

    if resolved == LivePreviewMethod.FULL:
        decoded = decode_latent_batch(processing.sd_model, denoised_latent)
        return _tensor_to_pil_rgb(decoded[0])

    logger.warning("Unknown live preview method '%s'; skipping preview.", method.value)
    return None


@dataclass(frozen=True)
class PreviewFactorsFit:
    model_name: str
    channels: int
    step: int
    total: int
    scale_h: int
    scale_w: int
    sample_count: int
    mse: float
    factors: tuple[tuple[float, float, float], ...]
    bias: tuple[float, float, float]
    vae_meta: dict[str, object]


def fit_preview_factors(
    processing: Any,
    denoised_latent: Any,
    *,
    step: int,
    total: int,
    sample_limit: int,
) -> Optional[PreviewFactorsFit]:
    import torch
    import torch.nn.functional as F

    from apps.backend.runtime.processing.conditioners import decode_latent_batch

    if not isinstance(denoised_latent, torch.Tensor) or denoised_latent.ndim != 4:
        return None

    try:
        decoded = decode_latent_batch(processing.sd_model, denoised_latent).detach().float()
    except Exception as exc:
        logger.warning("[preview-factors] decode_first_stage failed: %s", exc)
        return None

    if decoded.ndim != 4 or decoded.shape[1] != 3:
        logger.warning("[preview-factors] unexpected decoded shape=%s; skipping.", tuple(decoded.shape))
        return None

    latent_h, latent_w = int(denoised_latent.shape[-2]), int(denoised_latent.shape[-1])
    if latent_h <= 0 or latent_w <= 0:
        return None

    decoded_small = F.interpolate(decoded, size=(latent_h, latent_w), mode="area")

    channels = int(denoised_latent.shape[1])
    latent = denoised_latent.detach().float()[0].movedim(0, -1).reshape(-1, channels)
    rgb = decoded_small[0].movedim(0, -1).reshape(-1, 3)

    n_pixels = int(latent.shape[0])
    if n_pixels <= 0:
        return None

    sample_n = min(int(sample_limit), n_pixels)
    idx = torch.linspace(0, n_pixels - 1, steps=sample_n, device=latent.device).long()
    latent_s = latent.index_select(0, idx)
    rgb_s = rgb.index_select(0, idx)

    ones = torch.ones((sample_n, 1), device=latent_s.device, dtype=latent_s.dtype)
    latent_aug = torch.cat([latent_s, ones], dim=1)

    try:
        sol = torch.linalg.lstsq(latent_aug, rgb_s).solution  # (C+1, 3)
    except Exception as exc:
        logger.warning("[preview-factors] lstsq failed: %s", exc)
        return None

    pred = latent_aug @ sol
    mse = (pred - rgb_s).pow(2).mean().item()

    factors_rows = tuple(tuple(float(v) for v in row) for row in sol[:-1].detach().cpu().tolist())
    bias_row = tuple(float(v) for v in sol[-1].detach().cpu().tolist())

    scale_h = int(round(decoded.shape[-2] / float(latent_h))) if latent_h else 0
    scale_w = int(round(decoded.shape[-1] / float(latent_w))) if latent_w else 0

    vae_meta: dict[str, object] = {}
    try:
        vae = getattr(getattr(processing.sd_model, "codex_objects", None), "vae", None)
        fs = getattr(vae, "first_stage_model", None)
        for key in ("scaling_factor", "shift_factor", "latents_mean", "latents_std"):
            if hasattr(fs, key):
                vae_meta[key] = getattr(fs, key)
    except Exception:
        vae_meta = {}

    return PreviewFactorsFit(
        model_name=type(processing.sd_model).__name__,
        channels=channels,
        step=int(step),
        total=int(total),
        scale_h=scale_h,
        scale_w=scale_w,
        sample_count=int(sample_n),
        mse=float(mse),
        factors=factors_rows,
        bias=bias_row,
        vae_meta=vae_meta or {},
    )


def maybe_log_preview_factors(processing: Any, denoised_latent: Any, *, step: int, total: int) -> None:
    global _DEBUG_PREVIEW_FACTORS_LAST_JOB_TS

    if not debug_preview_factors_enabled():
        return

    job_ts = str(getattr(backend_state, "job_timestamp", "") or "")
    if job_ts and _DEBUG_PREVIEW_FACTORS_LAST_JOB_TS == job_ts:
        return

    fit = fit_preview_factors(
        processing,
        denoised_latent,
        step=int(step),
        total=int(total),
        sample_limit=debug_preview_factors_sample_limit(),
    )
    if fit is None:
        return

    logger.info(
        "[preview-factors] model=%s channels=%d step=%d/%d scale=%dx%d samples=%d mse=%.6g vae=%s",
        fit.model_name,
        fit.channels,
        fit.step,
        fit.total,
        fit.scale_h,
        fit.scale_w,
        fit.sample_count,
        fit.mse,
        fit.vae_meta,
    )
    logger.info("[preview-factors] factors = %s", tuple(tuple(round(v, 6) for v in row) for row in fit.factors))
    logger.info("[preview-factors] bias = %s", tuple(round(v, 6) for v in fit.bias))

    _DEBUG_PREVIEW_FACTORS_LAST_JOB_TS = job_ts or "(unknown)"


__all__ = [
    "LivePreviewMethod",
    "PreviewFactorsFit",
    "debug_preview_factors_enabled",
    "debug_preview_factors_sample_limit",
    "decode_preview_image",
    "fit_preview_factors",
    "live_preview_method_from_env",
    "live_preview_method_to_env",
    "maybe_log_preview_factors",
]
