"""Neural network building blocks for backend runtimes."""

from .base import ModuleDict, ObjectDict, Dummy
from .clip import IntegratedCLIP
from .t5 import IntegratedT5
from .unet import (
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
from .vae import AutoencoderKLWan
from .flux import (
    FluxTransformer2DModel,
    attention,
    rope,
    timestep_embedding as flux_timestep_embedding,
    EmbedND,
    MLPEmbedder,
    RMSNorm,
    QKNorm,
    SelfAttention,
)
from .mmditx import SD3Transformer2DModel
from .chroma import ChromaTransformer2DModel
from .cnets import cldm, t2i_adapter

__all__ = [
    "AutoencoderKLWan",
    "ChromaTransformer2DModel",
    "Dummy",
    "EmbedND",
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

# Compatibility alias: allow absolute import of wan_latent_norms when older paths were cached
# This keeps runtime resilient without introducing legacy shims elsewhere.
try:
    import importlib, sys as _sys
    _wl = importlib.import_module('.wan_latent_norms', __name__)
    _sys.modules[__name__ + '.wan_latent_norms'] = _wl
except Exception:
    pass
