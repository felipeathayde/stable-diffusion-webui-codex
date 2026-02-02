"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Init-image preparation helpers for img2img-style pipelines.
Encodes an init image into tensors/latents and returns a structured `InitImageBundle` for downstream pipelines.

Symbols (top-level; keep in sync; no ghosts):
- `prepare_init_bundle` (function): Converts a processing init image into a tensor/latent bundle (optionally includes a mask).
- `__all__` (constant): Explicit export list for the module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from apps.backend.core import devices
from apps.backend.runtime.processing.datatypes import InitImageBundle


def prepare_init_bundle(processing: Any) -> InitImageBundle:
    """Encode the init image into tensor/latents for downstream pipelines."""
    image = getattr(processing, "init_image", None)
    if image is None:
        raise ValueError("img2img requires processing.init_image")

    array = np.array(image).astype(np.float32) / 255.0
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
