"""Helpers for preparing init-image bundles for img2img-style workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from apps.backend.core import devices
from apps.backend.runtime.processing.datatypes import InitImageBundle


def prepare_init_bundle(processing: Any) -> InitImageBundle:
    """Encode the init image into tensor/latents for downstream workflows."""
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
