"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Convenience re-exports for neural network building blocks used by backend runtimes.
This module groups commonly used classes/functions so engines can import from a single location, while keeping model-specific logic in their own packages.

Symbols (top-level; keep in sync; no ghosts):
- `ModuleDict` (class): `torch.nn.Module` wrapper that registers a provided module dict (compat helper).
- `ObjectDict` (class): Attribute-access wrapper for a plain mapping (compat helper).
- `Dummy` (class): Minimal `ConfigMixin` module used by some diffusers-style patterns.
- `IntegratedCLIP` (class): Integrated CLIP wrapper exposing expected `logit_scale`/projection fields.
- `IntegratedT5` (class): Integrated T5 wrapper used by native pipelines.
- `UNet2DConditionModel` (class): Native UNet implementation used by SD-family engines.
- `FluxTransformer2DModel` (class): Native Flux transformer core.
- `ChromaTransformer2DModel` (class): Native Chroma transformer core.
- `AutoencoderKLWan` (class): WAN22-specific VAE implementation.
- `SD3Transformer2DModel` (class): SD3 transformer core.
- `__all__` (constant): Export list for the re-exported symbols.
"""

from apps.backend.runtime.common.nn.base import ModuleDict, ObjectDict, Dummy
from apps.backend.runtime.common.nn.clip import IntegratedCLIP
from apps.backend.runtime.common.nn.t5 import IntegratedT5
from apps.backend.runtime.common.nn.unet import (
    UNet2DConditionModel,
    conv_nd,
    avg_pool_nd,
    ResBlock,
    SpatialTransformer,
    TimestepEmbedSequential,
    timestep_embedding,
    exists,
    Downsample,
)
from apps.backend.runtime.wan22.vae import AutoencoderKLWan
from apps.backend.runtime.flux import FluxTransformer2DModel, FluxArchitectureConfig, FluxGuidanceConfig, FluxPositionalConfig
from apps.backend.runtime.flux.flux import (
    attention,
    rope,
    timestep_embedding as flux_timestep_embedding,
    EmbedND,
    MLPEmbedder,
    RMSNorm,
    QKNorm,
    SelfAttention,
)
from apps.backend.runtime.sd.mmditx import SD3Transformer2DModel
from apps.backend.runtime.chroma.chroma import ChromaTransformer2DModel
from apps.backend.runtime.sd.cnets import cldm, t2i_adapter

__all__ = [
    "AutoencoderKLWan",
    "ChromaTransformer2DModel",
    "Dummy",
    "EmbedND",
    "FluxArchitectureConfig",
    "FluxGuidanceConfig",
    "FluxPositionalConfig",
    "FluxTransformer2DModel",
    "IntegratedCLIP",
    "IntegratedT5",
    "MLPEmbedder",
    "ModuleDict",
    "ObjectDict",
    "QKNorm",
    "ResBlock",
    "RMSNorm",
    "SD3Transformer2DModel",
    "SelfAttention",
    "SpatialTransformer",
    "TimestepEmbedSequential",
    "UNet2DConditionModel",
    "avg_pool_nd",
    "attention",
    "cldm",
    "conv_nd",
    "Downsample",
    "exists",
    "flux_timestep_embedding",
    "rope",
    "t2i_adapter",
]
