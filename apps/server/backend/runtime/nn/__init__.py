"""Neural network modules used by backend runtimes."""

from .base import ModuleDict, ObjectDict, Dummy
from .clip import IntegratedCLIP
from .t5 import IntegratedT5
# Align export names: our implementation class is IntegratedUNet2DConditionModel
# but many call-sites expect UNet2DConditionModel for SD1/SDXL.
from .unet import (
    IntegratedUNet2DConditionModel as UNet2DConditionModel,
    conv_nd,
    avg_pool_nd,
    ResBlock,
    SpatialTransformer,
    TimestepEmbedSequential,
    timestep_embedding,
    exists,
    Downsample,
)
# Our VAE implementation is IntegratedAutoencoderKL; export it and an alias
# AutoencoderKLWan for compatibility with older codepaths expecting that name.
from .vae import IntegratedAutoencoderKL
from .vae import IntegratedAutoencoderKL as AutoencoderKLWan
from .flux import (
    IntegratedFluxTransformer2DModel as FluxTransformer2DModel,
    attention,
    rope,
    timestep_embedding as flux_timestep_embedding,
    EmbedND,
    MLPEmbedder,
    RMSNorm,
    QKNorm,
    SelfAttention,
)
# mmditx module defines MMDiTX; export a compatibility alias
from .mmditx import MMDiTX as MMDiTXTransformer3DModel
# chroma module defines IntegratedChromaTransformer2DModel; provide alias
from .chroma import IntegratedChromaTransformer2DModel as ChromaTransformer2DModel
from .cnets import cldm, t2i_adapter

__all__ = [
    "AutoencoderKLWan",
    "IntegratedAutoencoderKL",
    "ChromaTransformer2DModel",
    "IntegratedChromaTransformer2DModel",
    "Dummy",
    "EmbedND",
    "FluxTransformer2DModel",
    "IntegratedFluxTransformer2DModel",
    "IntegratedCLIP",
    "IntegratedT5",
    "MMDiTXTransformer3DModel",
    "MMDiTX",
    "MLPEmbedder",
    "ModuleDict",
    "ObjectDict",
    "QKNorm",
    "ResBlock",
    "RMSNorm",
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
