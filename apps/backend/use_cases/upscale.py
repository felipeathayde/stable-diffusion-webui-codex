"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Standalone upscale use-case (Option A).
Implements a strict, single-image upscale pipeline:
decode input → run selected upscaler (tiled, with optional OOM fallback) → return PIL image.

Symbols (top-level; keep in sync; no ghosts):
- `UpscaleParams` (dataclass): Validated upscale request params.
- `upscale_pil_image` (function): Upscales one RGB PIL image and returns the upscaled RGB PIL image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image

from apps.backend.runtime.vision.upscalers.registry import upscale_image_tensor
from apps.backend.runtime.vision.upscalers.specs import TileConfig, tile_config_from_payload


@dataclass(slots=True)
class UpscaleParams:
    upscaler_id: str
    scale: float
    tile: TileConfig

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "UpscaleParams":
        raw_id = payload.get("upscaler_id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError("Missing or invalid 'upscaler_id'")

        scale_raw = payload.get("scale", 2)
        try:
            scale = float(scale_raw)
        except Exception:
            raise ValueError("Invalid 'scale' (must be number)") from None
        if not (scale > 0):
            raise ValueError("Invalid 'scale' (must be > 0)")

        tile_cfg = tile_config_from_payload(payload.get("tile"), context="tile")

        return cls(upscaler_id=raw_id.strip(), scale=scale, tile=tile_cfg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "upscaler_id": self.upscaler_id,
            "scale": float(self.scale),
            "tile": {
                "tile": int(self.tile.tile),
                "overlap": int(self.tile.overlap),
                "fallback_on_oom": bool(self.tile.fallback_on_oom),
                "min_tile": int(self.tile.min_tile),
            },
        }


def _pil_to_tensor_01_rgb(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    arr = np.moveaxis(arr, 2, 0)
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor.to(dtype=torch.float32)


def _tensor_01_rgb_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3:
        raise ValueError(f"expected CHW tensor; got shape={tuple(image.shape)}")
    arr = image.detach().float().cpu().clamp(0, 1)
    arr = (arr * 255.0).round().byte().movedim(0, -1).numpy()
    return Image.fromarray(arr, mode="RGB")


@torch.inference_mode()
def upscale_pil_image(
    image: Image.Image,
    *,
    params: UpscaleParams,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Image.Image:
    base = image.convert("RGB")
    w, h = base.size
    target_w = max(1, int(round(float(w) * float(params.scale))))
    target_h = max(1, int(round(float(h) * float(params.scale))))

    tensor = _pil_to_tensor_01_rgb(base)
    out = upscale_image_tensor(
        tensor,
        upscaler_id=params.upscaler_id,
        target_width=target_w,
        target_height=target_h,
        tile=params.tile,
        progress_callback=progress_callback,
    )
    return _tensor_01_rgb_to_pil(out[0])


__all__ = ["UpscaleParams", "upscale_pil_image"]
