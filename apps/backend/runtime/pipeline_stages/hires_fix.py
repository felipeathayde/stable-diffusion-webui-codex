"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Global hires-fix pipeline stage (second pass orchestration helpers).
Provides shared helpers to prepare hires inputs (init latents + image-conditioning) using the global upscalers runtime,
and to compute correct `start_at_step` semantics from `denoise`.

Symbols (top-level; keep in sync; no ghosts):
- `start_at_step_from_denoise` (function): Maps `denoise` in [0..1] to `start_at_step` (0..steps-1) with correct monotonic semantics.
- `prepare_hires_latents_and_conditioning` (function): Prepares hires init latents + image-conditioning (SD-family v1).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import torch

from apps.backend.runtime.processing.datatypes import HiResPlan
from apps.backend.runtime.vision.upscalers.specs import TileConfig, default_tile_config


def start_at_step_from_denoise(*, denoise: float, steps: int) -> int:
    """Convert denoise strength to `start_at_step` with correct semantics.

    In this codebase, `start_at_step` controls how much of the init latent is preserved:
    - `start_at_step=0` → behaves like a full denoise (strong deviation).
    - `start_at_step=steps-1` → behaves like a near-no-op (minimal deviation).

    Therefore:
    - denoise=1 → start_at_step=0
    - denoise=0 → start_at_step=steps-1
    """

    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    d = float(denoise)
    if not math.isfinite(d):
        raise ValueError("denoise must be a finite number")
    if d < 0.0 or d > 1.0:
        raise ValueError("denoise must be in [0..1]")

    raw = int(round((1.0 - d) * float(steps)))
    return max(0, min(raw, int(steps) - 1))


def prepare_hires_latents_and_conditioning(
    processing: Any,
    *,
    base_samples: torch.Tensor,
    base_decoded: torch.Tensor | None,
    hires_plan: HiResPlan,
    tile: TileConfig | None = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare hires init latents + image-conditioning.

    v1: SD-family only (SD/SDXL). Other families should implement their own backend and be wired here.
    """

    if tile is None:
        tile = default_tile_config()

    sd_model = getattr(processing, "sd_model", None)
    if sd_model is None:
        raise ValueError("processing.sd_model is required for hires")

    from apps.backend.runtime.families.sd.hires_fix import prepare_hires_latents_and_conditioning as _sd_prepare

    return _sd_prepare(
        sd_model,
        base_samples=base_samples,
        base_decoded=base_decoded,
        target_width=int(hires_plan.target_width),
        target_height=int(hires_plan.target_height),
        upscaler_id=str(hires_plan.upscaler_id),
        tile=tile,
        image_mask=getattr(processing, "image_mask", None),
        round_mask=bool(getattr(processing, "round_image_mask", True)),
        progress_callback=progress_callback,
    )


__all__ = ["start_at_step_from_denoise", "prepare_hires_latents_and_conditioning"]
