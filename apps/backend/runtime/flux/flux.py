"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Compatibility surface re-exporting Flux runtime symbols and helper functions.

Symbols (top-level; keep in sync; no ghosts):
- `FluxArchitectureConfig` (dataclass): Transformer architecture config (dims/blocks/positional/guidance).
- `FluxGuidanceConfig` (dataclass): Optional guidance embedding config.
- `FluxPositionalConfig` (dataclass): Positional embedding config for Flux-like DiT models.
- `FluxTransformer2DModel` (class): Codex-native Flux transformer implementation.
- `EmbedND` (class): Builds N-D rotary frequency embeddings from integer positional IDs.
- `MLPEmbedder` (class): Two-layer MLP used for time/guidance/vector projections.
- `RMSNorm` (class): RMSNorm layer used by Flux components.
- `QKNorm` (class): Q/K normalization layer used by attention blocks.
- `DoubleStreamBlock` (class): Flux double-stream transformer block (dual-stream img+txt).
- `SingleStreamBlock` (class): Flux single-stream transformer block (concatenated tokens).
- `LastLayer` (class): Final projection layer for Flux transformer outputs.
- `SelfAttention` (class): Self-attention layer used inside Flux blocks.
- `attention` (function): Apply RoPE and delegate attention to the configured backend attention function.
- `rope` (function): Alias for `build_rotary_frequencies`.
- `apply_rope` (function): Alias for `apply_rotary_embeddings`.
- `timestep_embedding` (function): Sinusoidal timestep embedding helper used by Flux variants.
"""

from .config import FluxArchitectureConfig, FluxGuidanceConfig, FluxPositionalConfig
from apps.backend.runtime.attention import attention_function

from .geometry import apply_rotary_embeddings, build_rotary_frequencies as rope, timestep_embedding
from .embed import EmbedND, MLPEmbedder
from .components import RMSNorm, QKNorm, DoubleStreamBlock, SingleStreamBlock, LastLayer, SelfAttention
from .model import FluxTransformer2DModel


def attention(q, k, v, pe):
    q, k = apply_rotary_embeddings(q, k, pe)
    return attention_function(q, k, v, q.shape[1], skip_reshape=True)


apply_rope = apply_rotary_embeddings

__all__ = [
    "FluxArchitectureConfig",
    "FluxGuidanceConfig",
    "FluxPositionalConfig",
    "FluxTransformer2DModel",
    "EmbedND",
    "MLPEmbedder",
    "RMSNorm",
    "QKNorm",
    "DoubleStreamBlock",
    "SingleStreamBlock",
    "LastLayer",
    "SelfAttention",
    "attention",
    "rope",
    "apply_rope",
    "timestep_embedding",
]
