"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Spandrel backend adapter (single import boundary).
Implements load + inference for Spandrel SR models while keeping Spandrel types isolated from the rest of the codebase.
Enforces the upscaler safeweights policy: when `CODEX_SAFE_WEIGHTS=1`, non-`.safetensors` model files are rejected before loading.

Symbols (top-level; keep in sync; no ghosts):
- `SpandrelModelHandle` (dataclass): Loaded Spandrel model handle (descriptor + scale + channels).
- `load_spandrel_model` (function): Loads a model from file with strict validation.
- `run_spandrel_upscale` (function): Runs SR with optional tiling and OOM fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole

from .errors import UpscalerLoadError, UpscalerRuntimeError
from .safeweights import safeweights_enabled
from .specs import TileConfig
from .tiled_scale import tiled_scale

# Spandrel is intentionally only imported here (see AGENTS.md).
from spandrel import ImageModelDescriptor, ModelLoader  # type: ignore


@dataclass(slots=True)
class SpandrelModelHandle:
    descriptor: ImageModelDescriptor
    scale: float
    input_channels: int
    output_channels: int


def load_spandrel_model(path: str | Path) -> SpandrelModelHandle:
    p = Path(str(path))
    if not p.is_file():
        raise UpscalerLoadError(f"Upscaler model file not found: {p}")

    if safeweights_enabled() and p.suffix.lower() != ".safetensors":
        raise UpscalerLoadError(
            f"Unsafe upscaler weights blocked by CODEX_SAFE_WEIGHTS=1 (expected .safetensors): {p}"
        )

    try:
        descriptor = ModelLoader().load_from_file(str(p)).eval()
    except Exception as exc:
        raise UpscalerLoadError(f"Failed to load upscaler model: {p} ({exc})") from exc

    if not isinstance(descriptor, ImageModelDescriptor):
        raise UpscalerLoadError("Upscale model must be a single-image model (Spandrel ImageModelDescriptor).")

    scale = float(getattr(descriptor, "scale", 0.0) or 0.0)
    if not (scale > 0):
        raise UpscalerLoadError(f"Upscaler model scale is missing/invalid: {scale!r}")

    input_channels = int(getattr(descriptor, "input_channels", 3) or 3)
    output_channels = int(getattr(descriptor, "output_channels", 3) or 3)
    if input_channels != 3 or output_channels != 3:
        raise UpscalerLoadError(
            f"Only RGB upscalers are supported in v1 (expected 3->3 channels; got {input_channels}->{output_channels})."
        )

    return SpandrelModelHandle(
        descriptor=descriptor,
        scale=scale,
        input_channels=input_channels,
        output_channels=output_channels,
    )


@torch.inference_mode()
def run_spandrel_upscale(
    handle: SpandrelModelHandle,
    image: torch.Tensor,
    *,
    tile: TileConfig,
    device: torch.device,
    dtype: torch.dtype,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"image must be BCHW; got shape={tuple(image.shape)}")
    if image.shape[1] != 3:
        raise ValueError(f"image must be RGB (C=3); got shape={tuple(image.shape)}")

    descriptor = handle.descriptor
    model = getattr(descriptor, "model", None)
    if model is None or not isinstance(model, torch.nn.Module):
        raise UpscalerRuntimeError("Spandrel descriptor missing `.model` module.")

    oom_exc = memory_management.manager.oom_exception

    offload_device = None
    try:
        offload_device = memory_management.manager.get_offload_device(DeviceRole.CORE)
    except Exception:
        offload_device = None

    # Move model and input to the requested device/dtype.
    try:
        model.to(device=device, dtype=dtype)
    except Exception as exc:
        raise UpscalerRuntimeError(f"Failed to move upscaler model to {device}/{dtype}: {exc}") from exc

    x = image.to(device=device, dtype=dtype)

    try:
        # Fast path: no tiling (tile <= 0).
        if int(tile.tile) <= 0:
            try:
                y = descriptor(x)
            except Exception as exc:
                raise UpscalerRuntimeError(f"Upscaler inference failed: {exc}") from exc
            return y.to(dtype=torch.float32)

        # Tiled path with optional OOM fallback.
        tile_size = int(tile.tile)
        min_tile = int(tile.min_tile)
        overlap = int(tile.overlap)

        while True:
            try:
                return tiled_scale(
                    x,
                    lambda t: descriptor(t),
                    tile_x=tile_size,
                    tile_y=tile_size,
                    overlap=overlap,
                    upscale_amount=float(handle.scale),
                    out_channels=3,
                    output_device=device,
                    progress_callback=progress_callback,
                ).to(dtype=torch.float32)
            except oom_exc as exc:
                if not tile.fallback_on_oom:
                    raise UpscalerRuntimeError(
                        f"OOM during tiled upscaling (tile={tile_size}, overlap={overlap}). "
                        "Enable fallback or choose a smaller tile."
                    ) from exc
                next_tile = tile_size // 2
                if next_tile < min_tile:
                    raise UpscalerRuntimeError(
                        f"OOM during tiled upscaling and tile fallback exhausted (tile would drop below {min_tile})."
                    ) from exc
                tile_size = next_tile
                try:
                    memory_management.manager.soft_empty_cache(force=True)
                except Exception:
                    pass
    finally:
        # Avoid leaving the SR model resident on the primary device. The memory manager
        # does not track upscaler models (v1), so we explicitly offload to the role's
        # configured offload device.
        try:
            if offload_device is not None and offload_device != device:
                model.to(device=offload_device, dtype=torch.float32)
        except Exception:
            pass
        try:
            memory_management.manager.soft_empty_cache(force=True)
        except Exception:
            pass


__all__ = ["SpandrelModelHandle", "load_spandrel_model", "run_spandrel_upscale"]
