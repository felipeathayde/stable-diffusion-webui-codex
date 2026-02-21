"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Init-image preparation helpers for img2img-style pipelines.
Encodes an init image into tensors/latents and returns a structured `InitImageBundle` for downstream pipelines.
When `processing.width/height` are provided, the init image is resized to target output resolution before VAE encode.

Symbols (top-level; keep in sync; no ghosts):
- `prepare_init_bundle` (function): Converts a processing init image into a tensor/latent bundle (optionally includes a mask).
- `__all__` (constant): Explicit export list for the module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from apps.backend.core import devices
from apps.backend.runtime.processing.datatypes import InitImageBundle


def prepare_init_bundle(processing: Any) -> InitImageBundle:
    """Encode the init image into tensor/latents for downstream pipelines."""
    image = getattr(processing, "init_image", None)
    if image is None:
        raise ValueError("img2img requires processing.init_image")

    if not isinstance(image, Image.Image):
        raise TypeError(
            "img2img requires processing.init_image to be a PIL.Image.Image; "
            f"got {type(image).__name__}."
        )

    target_width = int(getattr(processing, "width", 0) or 0)
    target_height = int(getattr(processing, "height", 0) or 0)

    prepared_image = image.convert("RGB")
    if target_width > 0 and target_height > 0 and prepared_image.size != (target_width, target_height):
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        prepared_image = prepared_image.resize((target_width, target_height), resample=resample)
    setattr(processing, "init_image", prepared_image)

    array = np.array(prepared_image).astype(np.float32) / 255.0
    array = array * 2.0 - 1.0
    array = np.moveaxis(array, 2, 0)
    tensor = torch.from_numpy(np.expand_dims(array, axis=0)).to(
        devices.default_device(), dtype=torch.float32
    )
    latents = processing.sd_model.encode_first_stage(tensor)
    bundle = InitImageBundle(
        tensor=tensor,
        latents=latents,
        mask=getattr(processing, "image_mask", None) or getattr(processing, "mask", None),
        mode="latent",
    )
    return bundle


__all__ = ["prepare_init_bundle"]
