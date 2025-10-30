"""Compatibility module exposing Flux runtime components."""

from .config import FluxArchitectureConfig, FluxGuidanceConfig, FluxPositionalConfig
from apps.backend.runtime.attention import attention_function

from .geometry import apply_rotary_embeddings, build_rotary_frequencies as rope, timestep_embedding
from .embed import EmbedND, MLPEmbedder
from .components import RMSNorm, QKNorm, DoubleStreamBlock, SingleStreamBlock, LastLayer
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
    "attention",
    "rope",
    "apply_rope",
    "timestep_embedding",
]
