"""WAN 2.2 engine runtime specification (analogous to Flux spec.py).

This module provides the abstraction layer between the engine and runtime,
mirroring the Flux pattern with centralized runtime assembly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from apps.backend.patchers.unet import UnetPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.runtime.modules.k_prediction import FlowMatchEulerPrediction
from apps.backend.runtime.text_processing.t5_engine import T5TextProcessingEngine

logger = logging.getLogger("backend.engines.wan22.spec")


@dataclass(frozen=True)
class WanTextPipelines:
    """Text processing pipelines for WAN (T5 only, no CLIP)."""

    t5_text: T5TextProcessingEngine


@dataclass
class WanEngineRuntime:
    """Runtime container for WAN engine components.

    Analogous to FluxEngineRuntime, holds the assembled components.
    """

    vae: VAE
    unet: UnetPatcher
    text: WanTextPipelines
    device: str = "cuda"
    dtype: str = "bf16"


@dataclass(frozen=True)
class WanEngineSpec:
    """Specification for WAN engine variants."""

    name: str
    default_steps: int = 20
    default_cfg_scale: float = 7.5
    flow_shift: float = 8.0

    def is_14b(self) -> bool:
        return "14b" in self.name.lower()


def _k_predictor(spec: WanEngineSpec) -> FlowMatchEulerPrediction:
    """Create flow-match prediction for WAN."""
    logger.debug("Using FlowMatch predictor for WAN %s", spec.name)
    return FlowMatchEulerPrediction(
        mu=spec.flow_shift,
    )


def assemble_wan_runtime(
    *,
    spec: WanEngineSpec,
    codex_components: Mapping[str, object],
    estimated_config: Any,
    device: str = "cuda",
    dtype: str = "bf16",
    embedding_dir: Optional[str] = None,
    emphasis_name: str = "Original",
) -> WanEngineRuntime:
    """Assemble WAN runtime from Codex components.

    Mirrors assemble_flux_runtime pattern.

    Args:
        spec: WAN engine specification.
        codex_components: Dict with 'transformer', 'vae', 'text_encoder', 'tokenizer'.
        estimated_config: Model configuration object.
        device: Target device.
        dtype: Target dtype.
        embedding_dir: Optional embeddings directory.
        emphasis_name: Emphasis style for T5.

    Returns:
        Assembled WanEngineRuntime.
    """
    logger.debug("Assembling WAN runtime: %s", spec.name)

    vae_model = codex_components.get("vae")
    if vae_model is None:
        raise ValueError("WAN runtime requires 'vae' component")
    vae = VAE(model=vae_model)

    transformer = codex_components.get("transformer")
    if transformer is None:
        raise ValueError("WAN runtime requires 'transformer' component")

    k_predictor = _k_predictor(spec)
    unet = UnetPatcher.from_model(
        model=transformer,
        diffusers_scheduler=None,
        k_predictor=k_predictor,
        config=estimated_config,
    )

    t5_encoder = codex_components.get("text_encoder")
    t5_tokenizer = codex_components.get("tokenizer")
    if t5_encoder is None or t5_tokenizer is None:
        raise ValueError("WAN runtime requires 'text_encoder' and 'tokenizer' components")

    t5_engine = T5TextProcessingEngine(
        text_encoder=t5_encoder,
        tokenizer=t5_tokenizer,
        emphasis_name=emphasis_name,
        min_length=1,
    )

    logger.info(
        "WAN runtime assembled: spec=%s device=%s dtype=%s",
        spec.name,
        device,
        dtype,
    )

    return WanEngineRuntime(
        vae=vae,
        unet=unet,
        text=WanTextPipelines(t5_text=t5_engine),
        device=device,
        dtype=dtype,
    )


WAN_14B_SPEC = WanEngineSpec(
    name="wan22_14b",
    default_steps=20,
    default_cfg_scale=7.5,
    flow_shift=8.0,
)

WAN_5B_SPEC = WanEngineSpec(
    name="wan22_5b",
    default_steps=16,
    default_cfg_scale=6.0,
    flow_shift=6.0,
)

__all__ = [
    "WanTextPipelines",
    "WanEngineRuntime",
    "WanEngineSpec",
    "assemble_wan_runtime",
    "WAN_14B_SPEC",
    "WAN_5B_SPEC",
]

