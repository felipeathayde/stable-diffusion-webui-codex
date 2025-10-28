from __future__ import annotations

from typing import Optional, Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def decode_latent_batch(sd_model: Any, latents: torch.Tensor, *, target_device=None) -> torch.Tensor:
    decoded = sd_model.decode_first_stage(latents)
    if target_device is not None:
        decoded = decoded.to(target_device)
    return decoded


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
        mask_tensor = _prepare_mask(image_mask, round_mask=round_mask)
        if mask_tensor is None:
            mask_tensor = torch.ones(1, 1, source_image.shape[-2], source_image.shape[-1], device=source_image.device, dtype=source_image.dtype)
        else:
            mask_tensor = mask_tensor.to(device=source_image.device, dtype=source_image.dtype)
        conditioning_image = source_image
        conditioning_image = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(conditioning_image))
        mask_tensor = F.interpolate(mask_tensor, size=latent_image.shape[-2:])
        mask_tensor = mask_tensor.expand(conditioning_image.shape[0], -1, -1, -1)
        return torch.cat([mask_tensor, conditioning_image], dim=1)

    return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)
