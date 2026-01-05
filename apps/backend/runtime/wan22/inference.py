"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared WAN22 checkpoint shape inference helpers (detector + runtime loader).
Centralizes patch embedding / head shape interpretation so model registry detectors and WAN22 runtime loaders don't drift.

Symbols (top-level; keep in sync; no ghosts):
- `infer_wan22_patch_embedding` (function): Infers `(channels_in, model_dim, patch_size)` from `patch_embedding.weight`/`patch_embed.weight` shapes (torch or GGUF layout).
- `infer_wan22_patch_size_and_in_channels` (function): Infers (patch_size, channels_in) from `patch_embedding.weight`/`patch_embed.weight` shapes (torch or GGUF layout).
- `infer_wan22_latent_channels` (function): Infers latent channels from `head.head.weight`/`head.weight` and patch volume.
- `infer_wan22_io_channels` (function): Convenience wrapper returning `(channels_in, channels_out, patch_size)`.
"""

from __future__ import annotations

from typing import Sequence


def _as_shape(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, tuple) and all(isinstance(x, int) for x in value):
        return tuple(int(x) for x in value)
    if isinstance(value, list) and all(isinstance(x, int) for x in value):
        return tuple(int(x) for x in value)
    try:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        return tuple(int(x) for x in shape)
    except Exception:
        return None


def infer_wan22_patch_size_and_in_channels(
    patch_weight_shape: Sequence[int] | None,
    *,
    default_patch_size: tuple[int, int, int] = (1, 2, 2),
    default_in_channels: int = 16,
) -> tuple[tuple[int, int, int], int]:
    """Infer (patch_size, in_channels) from a Conv3d weight shape.

    Supports both:
    - torch layout: (out, in, kT, kH, kW)
    - GGUF layout:  (kT, kH, kW, in, out)
    """

    s = _as_shape(patch_weight_shape)
    if not s or len(s) != 5:
        return default_patch_size, int(default_in_channels)

    # torch layout is common in safetensors/pytorch exports.
    if s[0] >= 512 and s[2] <= 16 and s[3] <= 16 and s[4] <= 16:
        return (int(s[2]), int(s[3]), int(s[4])), int(s[1])

    # GGUF layout is used by some GGUF state dict loaders.
    if s[4] >= 512 and s[0] <= 16 and s[1] <= 16 and s[2] <= 16:
        return (int(s[0]), int(s[1]), int(s[2])), int(s[3])

    # Fallback: choose the side whose "out" dim looks like the WAN d_model (~5120).
    if s[0] >= s[4]:
        return (int(s[2]), int(s[3]), int(s[4])), int(s[1])
    return (int(s[0]), int(s[1]), int(s[2])), int(s[3])


def infer_wan22_patch_embedding(
    patch_weight_shape: Sequence[int] | None,
    *,
    default_patch_size: tuple[int, int, int] = (1, 2, 2),
    default_in_channels: int = 16,
    default_model_dim: int = 5120,
) -> tuple[int, int, tuple[int, int, int]]:
    """Infer `(in_channels, model_dim, patch_size)` from a Conv3d weight shape."""

    s = _as_shape(patch_weight_shape)
    if not s or len(s) != 5:
        return int(default_in_channels), int(default_model_dim), default_patch_size

    # torch layout: (out, in, kT, kH, kW)
    if s[0] >= 512 and s[2] <= 16 and s[3] <= 16 and s[4] <= 16:
        return int(s[1]), int(s[0]), (int(s[2]), int(s[3]), int(s[4]))

    # GGUF layout: (kT, kH, kW, in, out)
    if s[4] >= 512 and s[0] <= 16 and s[1] <= 16 and s[2] <= 16:
        return int(s[3]), int(s[4]), (int(s[0]), int(s[1]), int(s[2]))

    # Fallback: whichever side has the larger "out" dim.
    if s[0] >= s[4]:
        return int(s[1]), int(s[0]), (int(s[2]), int(s[3]), int(s[4]))
    return int(s[3]), int(s[4]), (int(s[0]), int(s[1]), int(s[2]))


def infer_wan22_latent_channels(
    head_weight_shape: Sequence[int] | None,
    *,
    patch_size: tuple[int, int, int],
    default_latent_channels: int,
) -> int:
    """Infer latent channels from head linear weight shape.

    `head.weight` is Linear(d_model -> patch_dim): shape (patch_dim, d_model).
    patch_dim = latent_channels * prod(patch_size)
    """

    s = _as_shape(head_weight_shape)
    if not s or len(s) != 2:
        return int(default_latent_channels)

    patch_dim = int(s[0])
    vol = int(patch_size[0] * patch_size[1] * patch_size[2])
    if vol <= 0:
        return int(default_latent_channels)
    if patch_dim % vol != 0:
        return int(default_latent_channels)

    latent = int(patch_dim // vol)
    return latent if latent > 0 else int(default_latent_channels)


def infer_wan22_io_channels(
    patch_weight_shape: Sequence[int] | None,
    head_weight_shape: Sequence[int] | None,
    *,
    default_patch_size: tuple[int, int, int] = (1, 2, 2),
    default_in_channels: int = 16,
) -> tuple[int, int, tuple[int, int, int]]:
    """Infer (channels_in, channels_out, patch_size) from patch+head shapes."""

    patch_size, channels_in = infer_wan22_patch_size_and_in_channels(
        patch_weight_shape,
        default_patch_size=default_patch_size,
        default_in_channels=default_in_channels,
    )
    channels_out = infer_wan22_latent_channels(
        head_weight_shape,
        patch_size=patch_size,
        default_latent_channels=channels_in,
    )
    return int(channels_in), int(channels_out), patch_size


__all__ = [
    "infer_wan22_io_channels",
    "infer_wan22_latent_channels",
    "infer_wan22_patch_embedding",
    "infer_wan22_patch_size_and_in_channels",
]
