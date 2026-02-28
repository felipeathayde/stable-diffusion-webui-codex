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
        tensor = mask.float()
    else:
        if isinstance(mask, Image.Image):
            array = np.array(mask.convert("L"), dtype=np.float32) / 255.0
        else:
            array = np.array(mask, dtype=np.float32)
        tensor = torch.from_numpy(array)

    if round_mask:
        tensor = (tensor > 0.5).to(dtype=torch.float32)
    else:
        tensor = tensor.to(dtype=torch.float32)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        raise ValueError(
            "image_mask must be 2D/3D/4D after normalization; "
            f"got ndim={tensor.ndim}."
        )
    return tensor


def encode_inpaint_latent_pair(
    sd_model: Any,
    source_image: torch.Tensor,
    *,
    image_mask: Optional[Any] = None,
    round_mask: bool = True,
    stage_prefix: str = "runtime.processing.conditioners.inpaint_pair",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode source + masked-conditioning images in a single VAE pass.

    Returns `(source_latent, conditioning_latent, conditioning_mask)`.
    """

    source_image = source_image.to(dtype=torch.float32)
    if source_image.ndim != 4:
        raise ValueError(f"source_image must be BCHW; got shape={tuple(source_image.shape)}")

    conditioning_mask = _prepare_mask(image_mask, round_mask=round_mask)
    if conditioning_mask is None:
        conditioning_mask = torch.ones(
            source_image.shape[0],
            1,
            source_image.shape[-2],
            source_image.shape[-1],
            device=source_image.device,
            dtype=source_image.dtype,
        )
    else:
        conditioning_mask = conditioning_mask.to(device=source_image.device, dtype=source_image.dtype)
        if conditioning_mask.shape[0] == 1 and source_image.shape[0] > 1:
            conditioning_mask = conditioning_mask.expand(source_image.shape[0], -1, -1, -1)
        elif conditioning_mask.shape[0] != source_image.shape[0]:
            raise ValueError(
                "image_mask batch size must be 1 or match source_image batch size; "
                f"got mask_batch={conditioning_mask.shape[0]} source_batch={source_image.shape[0]}."
            )

    conditioning_image = torch.lerp(
        source_image,
        source_image * (1.0 - conditioning_mask),
        1.0,
    )
    paired = torch.cat([source_image, conditioning_image], dim=0)
    paired_latents = sd_model.get_first_stage_encoding(
        encode_image_batch(
            sd_model,
            paired,
            stage=f"{stage_prefix}.encode_pair",
        )
    )
    batch = int(source_image.shape[0])
    if paired_latents.shape[0] != batch * 2:
        raise RuntimeError(
            "encode_inpaint_latent_pair received unexpected latent batch shape "
            f"{tuple(paired_latents.shape)} (expected first dimension {batch * 2})."
        )
    source_latent = paired_latents[:batch]
    conditioning_latent = paired_latents[batch:]
    return source_latent, conditioning_latent, conditioning_mask


def img2img_conditioning(
    sd_model: Any,
    source_image: torch.Tensor,
    latent_image: torch.Tensor,
    *,
    image_mask: Optional[Any] = None,
    round_mask: bool = True,
    precomputed_conditioning_latent: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
            if conditioning_mask.shape[0] == 1 and source_image.shape[0] > 1:
                conditioning_mask = conditioning_mask.expand(source_image.shape[0], -1, -1, -1)
            elif conditioning_mask.shape[0] != source_image.shape[0]:
                raise ValueError(
                    "image_mask batch size must be 1 or match source_image batch size; "
                    f"got mask_batch={conditioning_mask.shape[0]} source_batch={source_image.shape[0]}."
                )

        if precomputed_conditioning_latent is None:
            # Forge/A1111 parity: conditioning image is masked-out
            # (default inpainting_mask_weight=1.0).
            conditioning_image = torch.lerp(
                source_image,
                source_image * (1.0 - conditioning_mask),
                1.0,
            )
            conditioning_latent = sd_model.get_first_stage_encoding(
                encode_image_batch(
                    sd_model,
                    conditioning_image,
                    stage="runtime.processing.conditioners.img2img_conditioning.encode",
                )
            )
        else:
            conditioning_latent = precomputed_conditioning_latent
            if not torch.is_tensor(conditioning_latent) or conditioning_latent.ndim != 4:
                raise ValueError(
                    "precomputed_conditioning_latent must be a 4D torch.Tensor "
                    f"(got {type(conditioning_latent).__name__}, ndim={getattr(conditioning_latent, 'ndim', None)})."
                )
            conditioning_latent = conditioning_latent.to(device=latent_image.device, dtype=latent_image.dtype)
            if conditioning_latent.shape[0] != latent_image.shape[0]:
                raise ValueError(
                    "precomputed_conditioning_latent batch size must match latent_image batch size "
                    f"(got {conditioning_latent.shape[0]} vs {latent_image.shape[0]})."
                )

        mask_tensor = F.interpolate(conditioning_mask, size=conditioning_latent.shape[-2:])
        mask_tensor = mask_tensor.expand(conditioning_latent.shape[0], -1, -1, -1)
        return torch.cat([mask_tensor, conditioning_latent], dim=1)

    return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)
