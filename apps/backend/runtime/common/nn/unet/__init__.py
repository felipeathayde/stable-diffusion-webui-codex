"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public UNet API re-exports (model/config/layers/utils) for stable imports across runtimes.

Symbols (top-level; keep in sync; no ghosts):
- `UNet2DConditionModel` (class): UNet with cross-attention conditioning used by SD-family pipelines.
- `UNetConfig` (dataclass): Typed config describing channel/block/attention layout for `UNet2DConditionModel`.
- `BasicTransformerBlock` (class): Transformer block used inside `SpatialTransformer`.
- `CrossAttention` (class): Cross-attention layer for context conditioning.
- `Downsample` (class): UNet downsampling layer.
- `FeedForward` (class): MLP block used in transformer layers.
- `GEGLU` (class): Gated GELU linear unit used in `FeedForward`.
- `ResBlock` (class): Residual convolution block used in UNet backbones.
- `SpatialTransformer` (class): Spatial transformer module for attention over image tokens.
- `Timestep` (class): Timestep embedding/projection module.
- `TimestepBlock` (class): Interface for timestep-conditioned modules.
- `TimestepEmbedSequential` (class): Sequential container that forwards timestep/context when supported.
- `Upsample` (class): UNet upsampling layer.
- `apply_control` (function): Apply a ControlNet residual to a tensor when present in a `control` dict.
- `avg_pool_nd` (function): Build an AvgPool layer for 1D/2D/3D tensors.
- `checkpoint` (function): Gradient checkpoint stub (raises `NotImplementedError` when enabled).
- `conv_nd` (function): Build a Conv layer for 2D/3D tensors.
- `default` (function): Return a fallback when the provided value is `None`.
- `exists` (function): True when a value is not `None`.
- `timestep_embedding` (function): Sinusoidal timestep embedding helper.
"""

from __future__ import annotations

from .config import UNetConfig
from .layers import (
    BasicTransformerBlock,
    CrossAttention,
    Downsample,
    FeedForward,
    GEGLU,
    ResBlock,
    SpatialTransformer,
    Timestep,
    TimestepBlock,
    TimestepEmbedSequential,
    Upsample,
)
from .model import UNet2DConditionModel
from .utils import (
    apply_control,
    avg_pool_nd,
    checkpoint,
    conv_nd,
    default,
    exists,
    timestep_embedding,
)

__all__ = [
    "UNet2DConditionModel",
    "UNetConfig",
    "BasicTransformerBlock",
    "CrossAttention",
    "Downsample",
    "FeedForward",
    "GEGLU",
    "ResBlock",
    "SpatialTransformer",
    "Timestep",
    "TimestepBlock",
    "TimestepEmbedSequential",
    "Upsample",
    "apply_control",
    "avg_pool_nd",
    "checkpoint",
    "conv_nd",
    "default",
    "exists",
    "timestep_embedding",
]
