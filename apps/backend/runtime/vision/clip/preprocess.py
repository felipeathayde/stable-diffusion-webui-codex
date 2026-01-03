"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Preprocessing helpers for CLIP vision encoders in backend runtime.
Normalizes a batch of images (expected in BHWC) to the encoder input format (BCHW) using resize/crop and mean/std stats.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger for clip vision preprocessing.
- `_ensure_mean_std` (function): Validates and converts normalization stats into a tensor.
- `preprocess_image` (function): Resizes/crops and normalizes an image batch for a given preprocess spec.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch

from .errors import ClipVisionInputError
from .specs import ClipVisionPreprocessSpec

logger = logging.getLogger("backend.runtime.vision.clip.preprocess")


def _ensure_mean_std(values: Iterable[float], *, expected: int) -> torch.Tensor:
    values_tuple = tuple(float(v) for v in values)
    if len(values_tuple) != expected:
        raise ClipVisionInputError(f"Expected {expected} normalization values, received {len(values_tuple)}.")
    return torch.tensor(values_tuple)


def preprocess_image(
    image: torch.Tensor,
    spec: ClipVisionPreprocessSpec,
    *,
    crop: bool = True,
) -> torch.Tensor:
    """Normalize a BCHW image tensor for clip vision encoders."""
    if image.ndim != 4:
        raise ClipVisionInputError(f"Expected image tensor with 4 dims (batch, height, width, channels); got {image.shape}.")
    if image.shape[-1] < 3:
        raise ClipVisionInputError("Clip vision encoder requires RGB channels; received fewer than 3.")

    device = image.device
    dtype = image.dtype
    logger.debug(
        "Preprocessing clip vision batch: shape=%s dtype=%s device=%s crop=%s",
        tuple(image.shape),
        dtype,
        device,
        crop,
    )
    image = image[..., :3]  # enforce RGB
    mean = _ensure_mean_std(spec.mean, expected=3).to(device=device, dtype=dtype)
    std = _ensure_mean_std(spec.std, expected=3).to(device=device, dtype=dtype)
    image = image.movedim(-1, 1)
    if not (image.shape[2] == spec.image_size and image.shape[3] == spec.image_size):
        if crop:
            scale = spec.image_size / min(image.shape[2], image.shape[3])
            target_h = round(scale * image.shape[2])
            target_w = round(scale * image.shape[3])
        else:
            target_h = target_w = spec.image_size
        image = torch.nn.functional.interpolate(
            image,
            size=(target_h, target_w),
            mode="bicubic",
            antialias=True,
        )
        h_start = max((image.shape[2] - spec.image_size) // 2, 0)
        w_start = max((image.shape[3] - spec.image_size) // 2, 0)
        image = image[
            :,
            :,
            h_start : h_start + spec.image_size,
            w_start : w_start + spec.image_size,
        ]
    image = torch.clip(255.0 * image, 0.0, 255.0).round() / 255.0
    image = (image - mean.view(3, 1, 1)) / std.view(3, 1, 1)
    return image
