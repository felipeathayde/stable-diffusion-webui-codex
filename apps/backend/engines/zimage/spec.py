"""Z Image engine specification (analogous to Flux spec.py)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from apps.backend.patchers.unet import UnetPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.model_registry.family_runtime import get_family_spec, FamilyRuntimeSpec
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
    """Specification for Z Image engine.
    
    This spec delegates to FamilyRuntimeSpec for default values,
    with optional per-variant overrides.
    """
    name: str = "zimage"
    family: ModelFamily = ModelFamily.QWEN_IMAGE
    
    # Optional overrides (if None, delegates to FamilyRuntimeSpec)
    _flow_shift_override: Optional[float] = field(default=None, repr=False)
    _default_steps_override: Optional[int] = field(default=None, repr=False)
    _default_cfg_override: Optional[float] = field(default=None, repr=False)
    
    def _get_family_spec(self) -> FamilyRuntimeSpec:
        """Get the FamilyRuntimeSpec for this engine."""
        return get_family_spec(self.family)
    
    @property
    def flow_shift(self) -> float:
        """Flow-match shift, delegating to FamilyRuntimeSpec if not overridden."""
        if self._flow_shift_override is not None:
            return self._flow_shift_override
        return self._get_family_spec().flow_shift or 3.0
    
    @property
    def default_steps(self) -> int:
        """Default sampling steps, delegating to FamilyRuntimeSpec if not overridden."""
        if self._default_steps_override is not None:
            return self._default_steps_override
        return self._get_family_spec().default_steps
    
    @property
    def default_cfg_scale(self) -> float:
        """Default CFG scale, delegating to FamilyRuntimeSpec if not overridden."""
        if self._default_cfg_override is not None:
            return self._default_cfg_override
        return self._get_family_spec().default_cfg


def _k_predictor(spec: ZImageEngineSpec) -> FlowMatchEulerPrediction:
    """Create flow-match predictor for Z Image."""
    logger.debug("Using FlowMatch predictor for Z Image (shift=%.2f)", spec.flow_shift)
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
    vae = VAE(model=vae_model, family=ModelFamily.QWEN_IMAGE)
    
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
