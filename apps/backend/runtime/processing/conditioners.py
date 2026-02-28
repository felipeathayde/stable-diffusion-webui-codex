"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Preprocessing helpers for building conditioning tensors and decoding latents.
Implements txt2img/img2img image-conditioning helpers (including inpaint mask handling) used by SD-family engines.

Symbols (top-level; keep in sync; no ghosts):
- `decode_latent_batch` (function): Decode a batch of latents via `sd_model.decode_first_stage`, with optional smart-offload pre-VAE guard.
- `encode_image_batch` (function): Encode a BCHW image tensor via `sd_model.encode_first_stage`, with optional smart-offload pre-VAE guard.
- `txt2img_conditioning` (function): Build txt2img image-conditioning tensor (inpaint vs non-inpaint layouts).
- `_prepare_mask` (function): Normalize mask inputs (tensor/PIL/array) into a batched float tensor (optionally rounded).
- `img2img_conditioning` (function): Build img2img image-conditioning tensor (Forge/A1111 parity: masks-out conditioning image for inpaint).
"""

from __future__ import annotations

from typing import Optional, Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from apps.backend.runtime.memory.smart_offload_invariants import enforce_smart_offload_pre_vae_residency

def decode_latent_batch(
    sd_model: Any,
    latents: torch.Tensor,
    *,
    target_device=None,
    stage: str | None = None,
) -> torch.Tensor:
    if isinstance(stage, str) and stage.strip():
        enforce_smart_offload_pre_vae_residency(sd_model, stage=stage.strip())
    decoded = sd_model.decode_first_stage(latents)
    if target_device is not None:
        decoded = decoded.to(target_device)
    return decoded


def encode_image_batch(
    sd_model: Any,
    images: torch.Tensor,
    *,
    target_device=None,
    stage: str | None = None,
) -> torch.Tensor:
    if isinstance(stage, str) and stage.strip():
        enforce_smart_offload_pre_vae_residency(sd_model, stage=stage.strip())
    latents = sd_model.encode_first_stage(images)
    if target_device is not None:
        latents = latents.to(target_device)
    return latents


def txt2img_conditioning(sd_model: Any, latents: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if getattr(sd_model, "is_inpaint", False):
        image_conditioning = torch.ones(latents.shape[0], 3, height, width, device=latents.device, dtype=latents.dtype)
        image_conditioning = image_conditioning * 0.5
        image_conditioning = image_conditioning.to(latents.dtype)
        return F.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0)
    return latents.new_zeros(latents.shape[0], 5, 1, 1)


def _prepare_mask(mask: Any, *, round_mask: bool = True) -> torch.Tensor:
    if mask is None:
        return mask
    if torch.is_tensor(mask):
        tensor = mask
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.float()
    if isinstance(mask, Image.Image):
        array = np.array(mask.convert("L"), dtype=np.float32) / 255.0
    else:
        array = np.array(mask, dtype=np.float32)
    if round_mask:
        array = (array > 0.5).astype(np.float32)
    tensor = torch.from_numpy(array)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.unsqueeze(0)


def img2img_conditioning(sd_model: Any, source_image: torch.Tensor, latent_image: torch.Tensor, *, image_mask: Optional[Any] = None, round_mask: bool = True) -> torch.Tensor:
    source_image = source_image.to(dtype=torch.float32)

    if getattr(sd_model, "is_inpaint", False):
        conditioning_mask = _prepare_mask(image_mask, round_mask=round_mask)
        if conditioning_mask is None:
            conditioning_mask = torch.ones(
                1,
                1,
                source_image.shape[-2],
                source_image.shape[-1],
                device=source_image.device,
                dtype=source_image.dtype,
            )
        else:
            conditioning_mask = conditioning_mask.to(device=source_image.device, dtype=source_image.dtype)

        # Forge/A1111 parity: conditioning image is masked-out (default inpainting_mask_weight=1.0).
        conditioning_image = torch.lerp(
            source_image,
            source_image * (1.0 - conditioning_mask),
            1.0,
        )
        conditioning_image = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(conditioning_image))

        mask_tensor = F.interpolate(conditioning_mask, size=latent_image.shape[-2:])
        mask_tensor = mask_tensor.expand(conditioning_image.shape[0], -1, -1, -1)
        return torch.cat([mask_tensor, conditioning_image], dim=1)

    return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)
