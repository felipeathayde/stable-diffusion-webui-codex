"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SD-family hires-fix helpers (second pass init preparation).
Prepares the hi-res pass init latents and image-conditioning, routing either:
- latent interpolation upscalers (`latent:*`), or
- external SR models via the global upscalers runtime (`spandrel:*`).

Symbols (top-level; keep in sync; no ghosts):
- `prepare_hires_latents_and_conditioning` (function): Build hires init latents + image-conditioning for SD/SDXL.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from apps.backend.runtime.processing.conditioners import (
    decode_latent_batch,
    img2img_conditioning,
    txt2img_conditioning,
)
from apps.backend.runtime.vision.upscalers.registry import upscale_image_tensor, upscale_latent_tensor
from apps.backend.runtime.vision.upscalers.specs import LATENT_UPSCALE_MODES, TileConfig


def prepare_hires_latents_and_conditioning(
    sd_model: Any,
    *,
    base_samples: torch.Tensor,
    base_decoded: torch.Tensor | None,
    target_width: int,
    target_height: int,
    upscaler_id: str,
    tile: TileConfig,
    image_mask: Any | None = None,
    round_mask: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare hires init latents + image-conditioning for SD-family pipelines."""

    if not isinstance(upscaler_id, str) or not upscaler_id.strip():
        raise ValueError("Missing hires upscaler id")

    uid = upscaler_id.strip()
    if int(target_width) <= 0 or int(target_height) <= 0:
        raise ValueError("target_width/target_height must be positive")

    # Latent upscalers: scale latents directly (no external model).
    if uid in LATENT_UPSCALE_MODES:
        latent_w = max(1, int(target_width) // 8)
        latent_h = max(1, int(target_height) // 8)
        latents = upscale_latent_tensor(
            base_samples,
            upscaler_id=uid,
            target_width=latent_w,
            target_height=latent_h,
        )

        if getattr(sd_model, "is_inpaint", False):
            tensor = decode_latent_batch(sd_model, latents)
            image_conditioning = img2img_conditioning(
                sd_model,
                tensor,
                latents,
                image_mask=image_mask,
                round_mask=round_mask,
            )
        else:
            image_conditioning = txt2img_conditioning(sd_model, latents, int(target_width), int(target_height))

        return latents, image_conditioning

    # External SR models (spandrel): decode → SR in pixel space → re-encode.
    if uid.startswith("spandrel:"):
        decoded = base_decoded
        if decoded is None:
            decoded = decode_latent_batch(sd_model, base_samples).to(dtype=torch.float32)
        else:
            decoded = decoded.to(dtype=torch.float32)

        # Convert decoded [-1..1] to pixel [0..1] for SR.
        pixel_01 = decoded.add(1.0).mul(0.5).clamp(0.0, 1.0)
        upscaled_01 = upscale_image_tensor(
            pixel_01,
            upscaler_id=uid,
            target_width=int(target_width),
            target_height=int(target_height),
            tile=tile,
            progress_callback=progress_callback,
        )

        # Convert back to [-1..1] for VAE encode and image-conditioning.
        tensor = upscaled_01.mul(2.0).sub(1.0)
        latents = sd_model.encode_first_stage(tensor)
        image_conditioning = img2img_conditioning(
            sd_model,
            tensor,
            latents,
            image_mask=image_mask,
            round_mask=round_mask,
        )
        return latents, image_conditioning

    raise ValueError(f"Unsupported hires upscaler id: {uid!r}")


__all__ = ["prepare_hires_latents_and_conditioning"]

