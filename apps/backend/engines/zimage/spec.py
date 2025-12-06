"""Z Image engine specification (analogous to Flux spec.py)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from apps.backend.patchers.unet import UnetPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.runtime.modules.k_prediction import FlowMatchEulerPrediction

logger = logging.getLogger("backend.engines.zimage.spec")


@dataclass(frozen=True)
class ZImageTextPipelines:
    """Text processing pipeline for Z Image."""
    qwen3_text: Any  # ZImageTextProcessingEngine


@dataclass
class ZImageEngineRuntime:
    """Runtime container for Z Image engine components."""
    vae: VAE
    unet: UnetPatcher  # wraps ZImageTransformer2DModel
    text: ZImageTextPipelines
    device: str = "cuda"
    dtype: str = "bf16"


@dataclass(frozen=True)
class ZImageEngineSpec:
    """Specification for Z Image engine."""
    name: str = "zimage"
    default_steps: int = 8  # Turbo model is fast
    default_cfg_scale: float = 4.0
    flow_shift: float = 3.0


def _k_predictor(spec: ZImageEngineSpec) -> FlowMatchEulerPrediction:
    """Create flow-match predictor for Z Image."""
    return FlowMatchEulerPrediction(mu=spec.flow_shift)


def assemble_zimage_runtime(
    *,
    spec: ZImageEngineSpec,
    codex_components: Mapping[str, object],
    estimated_config: Any,
    device: str = "cuda",
    dtype: str = "bf16",
) -> ZImageEngineRuntime:
    """Assemble Z Image runtime from components.
    
    Args:
        spec: Engine specification.
        codex_components: Dict with 'transformer', 'vae', 'text_encoder'.
        estimated_config: Model config.
        device: Target device.
        dtype: Target dtype.
    
    Returns:
        Assembled ZImageEngineRuntime.
    """
    logger.debug("Assembling Z Image runtime")
    
    # VAE
    vae_model = codex_components.get("vae")
    if vae_model is None:
        raise ValueError("Z Image requires 'vae' component")
    vae = VAE(model=vae_model)
    
    # Transformer -> UnetPatcher
    transformer = codex_components.get("transformer")
    if transformer is None:
        raise ValueError("Z Image requires 'transformer' component")
    
    k_predictor = _k_predictor(spec)
    unet = UnetPatcher.from_model(
        model=transformer,
        diffusers_scheduler=None,
        k_predictor=k_predictor,
        config=estimated_config,
    )
    
    # Text encoder
    text_encoder = codex_components.get("text_encoder")
    if text_encoder is None:
        raise ValueError("Z Image requires 'text_encoder' component")
    
    from apps.backend.runtime.zimage.text_encoder import ZImageTextProcessingEngine
    text_engine = ZImageTextProcessingEngine(text_encoder)
    
    logger.info("Z Image runtime assembled: device=%s dtype=%s", device, dtype)
    
    return ZImageEngineRuntime(
        vae=vae,
        unet=unet,
        text=ZImageTextPipelines(qwen3_text=text_engine),
        device=device,
        dtype=dtype,
    )


# Default spec
ZIMAGE_SPEC = ZImageEngineSpec()

__all__ = [
    "ZImageTextPipelines",
    "ZImageEngineRuntime",
    "ZImageEngineSpec",
    "assemble_zimage_runtime",
    "ZIMAGE_SPEC",
]
