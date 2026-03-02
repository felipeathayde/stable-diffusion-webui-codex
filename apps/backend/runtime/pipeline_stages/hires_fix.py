"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Global hires-fix pipeline stage (second pass orchestration helpers).
Provides family-dispatched helpers to prepare hires inputs (latents/image-conditioning + continuation mode) using the
global upscalers runtime, and computes correct `start_at_step` semantics from `denoise`. Also computes a Forge-like
“fill then crop” resize plan when `resize_x/resize_y` change the aspect ratio (avoid stretching).

Symbols (top-level; keep in sync; no ghosts):
- `HiresFillCropPlan` (dataclass): Aspect-preserving hires resize plan (internal fill size + crop offsets).
- `HiresPreparation` (dataclass): Prepared hires inputs and continuation mode for the second pass (`init_latent` or `image_latents`).
- `compute_hires_fill_crop_plan` (function): Compute a fill-then-crop plan for hires pass (Forge-like semantics).
- `start_at_step_from_denoise` (function): Maps `denoise` in [0..1] to `start_at_step` (0..steps-1) with correct monotonic semantics.
- `prepare_hires_latents_and_conditioning` (function): Prepares hires inputs via family-dispatched backends (SD, flow-like, Kontext).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal, Optional

import torch

from apps.backend.runtime.processing.conditioners import decode_latent_batch, encode_image_batch
from apps.backend.runtime.processing.datatypes import HiResPlan
from apps.backend.runtime.vision.upscalers.registry import upscale_image_tensor, upscale_latent_tensor
from apps.backend.runtime.vision.upscalers.specs import LATENT_UPSCALE_MODES, TileConfig, default_tile_config


@dataclass(frozen=True, slots=True)
class HiresFillCropPlan:
    """Aspect-preserving hires resize plan (fill then crop)."""

    base_width: int
    base_height: int
    target_width: int
    target_height: int
    internal_width: int
    internal_height: int
    crop_left: int
    crop_top: int

    def needs_crop(self) -> bool:
        return (
            self.internal_width != self.target_width
            or self.internal_height != self.target_height
            or self.crop_left != 0
            or self.crop_top != 0
        )


HiresContinuationMode = Literal["init_latent", "image_latents"]
HiresBackend = Literal["sd", "flow", "kontext"]


@dataclass(frozen=True, slots=True)
class HiresPreparation:
    """Prepared hires tensors and continuation semantics for the second pass."""

    latents: torch.Tensor
    image_conditioning: torch.Tensor | None
    continuation_mode: HiresContinuationMode


def _ceil_div(num: int, den: int) -> int:
    if den <= 0:
        raise ValueError("den must be positive")
    if num < 0:
        raise ValueError("num must be >= 0")
    return (num + den - 1) // den


def compute_hires_fill_crop_plan(
    *,
    base_width: int,
    base_height: int,
    target_width: int,
    target_height: int,
) -> HiresFillCropPlan:
    """Compute an aspect-preserving hires resize plan (fill then crop; Forge-like semantics).

    Notes:
    - All dimensions are in pixels.
    - This function operates in latent-grid units (8px) to keep internal sizes aligned with VAE latents.
    - Crop offsets are in pixels and may be non-multiples of 8 (crop happens in pixel space).
    """

    bw = int(base_width)
    bh = int(base_height)
    tw = int(target_width)
    th = int(target_height)

    if bw <= 0 or bh <= 0:
        raise ValueError("base_width/base_height must be positive")
    if tw <= 0 or th <= 0:
        raise ValueError("target_width/target_height must be positive")
    if bw % 8 != 0 or bh % 8 != 0:
        raise ValueError("base_width/base_height must be multiples of 8")
    if tw % 8 != 0 or th % 8 != 0:
        raise ValueError("target_width/target_height must be multiples of 8")

    bw_l = bw // 8
    bh_l = bh // 8
    tw_l = tw // 8
    th_l = th // 8

    if bw_l <= 0 or bh_l <= 0:
        raise ValueError("Invalid base latent dimensions")
    if tw_l <= 0 or th_l <= 0:
        raise ValueError("Invalid target latent dimensions")

    # Compare aspect ratios using integers:
    # base wider-than-target iff bw/bh >= tw/th  <=>  bw*th >= bh*tw.
    left = bw_l * th_l
    right = bh_l * tw_l

    internal_w_l = tw_l
    internal_h_l = th_l

    if left == right:
        internal_w_l = tw_l
        internal_h_l = th_l
    elif left > right:
        # Base is wider: preserve aspect by matching target height and expanding width.
        internal_h_l = th_l
        internal_w_l = _ceil_div(th_l * bw_l, bh_l)
    else:
        # Base is taller: preserve aspect by matching target width and expanding height.
        internal_w_l = tw_l
        internal_h_l = _ceil_div(tw_l * bh_l, bw_l)

    internal_w = int(internal_w_l) * 8
    internal_h = int(internal_h_l) * 8

    if internal_w < tw or internal_h < th:
        raise ValueError("Internal hires size must cover the target size (internal < target)")

    crop_left = max(0, (internal_w - tw) // 2)
    crop_top = max(0, (internal_h - th) // 2)
    if crop_left + tw > internal_w or crop_top + th > internal_h:
        raise ValueError("Invalid crop plan (crop exceeds internal bounds)")

    return HiresFillCropPlan(
        base_width=bw,
        base_height=bh,
        target_width=tw,
        target_height=th,
        internal_width=internal_w,
        internal_height=internal_h,
        crop_left=int(crop_left),
        crop_top=int(crop_top),
    )


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


def _resolve_hires_backend(engine_id: str) -> HiresBackend:
    normalized_engine_id = str(engine_id or "").strip()
    if normalized_engine_id == "":
        raise NotImplementedError("Hires preparation requires a non-empty engine id.")
    if normalized_engine_id in {"sd15", "sd20", "sdxl", "sdxl_refiner", "sd35"}:
        return "sd"
    if normalized_engine_id in {"flux1", "flux1_fill", "flux1_chroma", "zimage", "anima"}:
        return "flow"
    if normalized_engine_id == "flux1_kontext":
        return "kontext"
    raise NotImplementedError(
        f"Hires preparation backend is not implemented for engine '{normalized_engine_id}'."
    )


def _prepare_flow_hires_latents(
    sd_model: Any,
    *,
    base_samples: torch.Tensor,
    base_decoded: torch.Tensor | None,
    upscaler_id: str,
    tile: TileConfig,
    progress_callback: Optional[Callable[[int, int], None]],
    resize_plan: HiresFillCropPlan,
) -> torch.Tensor:
    if not isinstance(upscaler_id, str) or not upscaler_id.strip():
        raise ValueError("Missing hires upscaler id")

    uid = upscaler_id.strip()
    target_width = int(resize_plan.target_width)
    target_height = int(resize_plan.target_height)
    internal_width = int(resize_plan.internal_width)
    internal_height = int(resize_plan.internal_height)
    crop_left = int(resize_plan.crop_left)
    crop_top = int(resize_plan.crop_top)

    if uid in LATENT_UPSCALE_MODES:
        internal_latent_width = max(1, internal_width // 8)
        internal_latent_height = max(1, internal_height // 8)
        upscaled_latents = upscale_latent_tensor(
            base_samples,
            upscaler_id=uid,
            target_width=internal_latent_width,
            target_height=internal_latent_height,
        )
        if internal_width == target_width and internal_height == target_height and crop_left == 0 and crop_top == 0:
            return upscaled_latents

        decoded = decode_latent_batch(
            sd_model,
            upscaled_latents,
            stage="hires.prepare.flow.latent.crop_decode",
        ).to(dtype=torch.float32)
        cropped = decoded[:, :, crop_top : crop_top + target_height, crop_left : crop_left + target_width]
        return encode_image_batch(
            sd_model,
            cropped,
            stage="hires.prepare.flow.latent.crop_encode",
        )

    if uid.startswith("spandrel:"):
        decoded = base_decoded
        if decoded is None:
            decoded = decode_latent_batch(
                sd_model,
                base_samples,
                stage="hires.prepare.flow.spandrel.decode_base",
            ).to(dtype=torch.float32)
        else:
            decoded = decoded.to(dtype=torch.float32)

        pixel_01 = decoded.add(1.0).mul(0.5).clamp(0.0, 1.0)
        upscaled_01 = upscale_image_tensor(
            pixel_01,
            upscaler_id=uid,
            target_width=internal_width,
            target_height=internal_height,
            tile=tile,
            progress_callback=progress_callback,
        )
        if internal_width != target_width or internal_height != target_height or crop_left != 0 or crop_top != 0:
            upscaled_01 = upscaled_01[:, :, crop_top : crop_top + target_height, crop_left : crop_left + target_width]

        tensor = upscaled_01.mul(2.0).sub(1.0)
        return encode_image_batch(
            sd_model,
            tensor,
            stage="hires.prepare.flow.spandrel.encode_upscaled",
        )

    raise ValueError(f"Unsupported hires upscaler id: {uid!r}")


def prepare_hires_latents_and_conditioning(
    processing: Any,
    *,
    base_samples: torch.Tensor,
    base_decoded: torch.Tensor | None,
    hires_plan: HiResPlan,
    tile: TileConfig | None = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> HiresPreparation:
    """Prepare hires inputs using family-dispatched backends."""

    if tile is None:
        tile = default_tile_config()

    sd_model = getattr(processing, "sd_model", None)
    if sd_model is None:
        raise ValueError("processing.sd_model is required for hires")

    base_latent_h = int(base_samples.shape[-2])
    base_latent_w = int(base_samples.shape[-1])
    resize_plan = compute_hires_fill_crop_plan(
        base_width=base_latent_w * 8,
        base_height=base_latent_h * 8,
        target_width=int(hires_plan.target_width),
        target_height=int(hires_plan.target_height),
    )
    if resize_plan.needs_crop():
        processing.update_extra_param(
            "Hires resize",
            f"fill-crop {resize_plan.internal_width}x{resize_plan.internal_height} -> {resize_plan.target_width}x{resize_plan.target_height}",
        )
        processing.update_extra_param("Hires crop left", int(resize_plan.crop_left))
        processing.update_extra_param("Hires crop top", int(resize_plan.crop_top))

    engine_id = str(getattr(sd_model, "engine_id", "") or "").strip()
    backend = _resolve_hires_backend(engine_id)

    if backend == "sd":
        from apps.backend.runtime.families.sd.hires_fix import prepare_hires_latents_and_conditioning as _sd_prepare

        latents, image_conditioning = _sd_prepare(
            sd_model,
            base_samples=base_samples,
            base_decoded=base_decoded,
            target_width=int(resize_plan.target_width),
            target_height=int(resize_plan.target_height),
            upscaler_id=str(hires_plan.upscaler_id),
            tile=tile,
            image_mask=getattr(processing, "image_mask", None),
            round_mask=bool(getattr(processing, "round_image_mask", True)),
            progress_callback=progress_callback,
            resize_plan=resize_plan,
        )
        return HiresPreparation(
            latents=latents,
            image_conditioning=image_conditioning,
            continuation_mode="init_latent",
        )

    if backend == "flow":
        latents = _prepare_flow_hires_latents(
            sd_model,
            base_samples=base_samples,
            base_decoded=base_decoded,
            upscaler_id=str(hires_plan.upscaler_id),
            tile=tile,
            progress_callback=progress_callback,
            resize_plan=resize_plan,
        )
        return HiresPreparation(
            latents=latents,
            image_conditioning=None,
            continuation_mode="init_latent",
        )

    if backend == "kontext":
        latents = _prepare_flow_hires_latents(
            sd_model,
            base_samples=base_samples,
            base_decoded=base_decoded,
            upscaler_id=str(hires_plan.upscaler_id),
            tile=tile,
            progress_callback=progress_callback,
            resize_plan=resize_plan,
        )
        return HiresPreparation(
            latents=latents,
            image_conditioning=None,
            continuation_mode="image_latents",
        )

    raise RuntimeError(f"Unsupported hires backend {backend!r} for engine_id={engine_id!r}.")


__all__ = [
    "HiresFillCropPlan",
    "HiresPreparation",
    "compute_hires_fill_crop_plan",
    "prepare_hires_latents_and_conditioning",
    "start_at_step_from_denoise",
]
