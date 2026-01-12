"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 engine runtime specification (analogous to Flux `spec.py`).
Defines the engine-facing `WanEngineSpec`/`WanEngineRuntime` containers and centralized runtime assembly used by WAN engines, delegating
defaults to `FamilyRuntimeSpec` and wiring the T5 text pipeline + denoiser/VAE patchers.

Symbols (top-level; keep in sync; no ghosts):
- `WanTextPipelines` (dataclass): Text processing pipelines for WAN (T5 only; no CLIP).
- `WanEngineRuntime` (dataclass): Runtime container for WAN components (VAE, denoiser patcher, text pipelines, device/dtype).
- `WanEngineSpec` (dataclass): Engine spec wrapper that delegates defaults to `FamilyRuntimeSpec` with per-variant overrides.
- `_k_predictor` (function): Builds the WAN flow predictor configured from the spec.
- `assemble_wan_runtime` (function): Assembles a `WanEngineRuntime` from a model family spec + loaded components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from apps.backend.patchers.denoiser import DenoiserPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.runtime.model_registry.flow_shift import flow_shift_spec_from_repo_dir
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.model_registry.family_runtime import get_family_spec, FamilyRuntimeSpec
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
    denoiser: DenoiserPatcher  # wraps WanTransformer2DModel
    text: WanTextPipelines
    device: str = "cuda"
    dtype: str = "bf16"


@dataclass(frozen=True)
class WanEngineSpec:
    """Specification for WAN engine variants.
    
    This spec delegates to FamilyRuntimeSpec for default values,
    with optional per-variant overrides.
    """
    name: str
    family: ModelFamily = ModelFamily.WAN22
    
    # Optional overrides (if None, delegates to FamilyRuntimeSpec)
    _flow_shift_override: Optional[float] = field(default=None, repr=False)
    _default_steps_override: Optional[int] = field(default=None, repr=False)
    _default_cfg_override: Optional[float] = field(default=None, repr=False)
    
    def _get_family_spec(self) -> FamilyRuntimeSpec:
        """Get the FamilyRuntimeSpec for this engine."""
        return get_family_spec(self.family)
    
    @property
    def flow_shift(self) -> float:
        """Flow-match shift (mu).

        Source of truth is the WAN diffusers `scheduler/scheduler_config.json` shipped
        with the official repos (vendored under `apps/backend/huggingface/Wan-AI/**`).
        This avoids hard-coded drift between 5B/14B variants and keeps parity with
        `.refs/diffusers` conversion scripts.
        """
        if self._flow_shift_override is not None:
            return self._flow_shift_override
        from apps.backend.engines.wan22.wan22_common import resolve_wan_repo_candidates
        from apps.backend.infra.config.repo_root import get_repo_root

        repo_root = get_repo_root()
        hf_root = repo_root / "apps" / "backend" / "huggingface"
        last_exc: Exception | None = None
        last_repo: str | None = None
        for rid in resolve_wan_repo_candidates(self.name):
            local_dir = hf_root / rid.replace("/", "/")
            try:
                spec = flow_shift_spec_from_repo_dir(local_dir)
                return spec.resolve()
            except Exception as exc:
                last_exc = exc
                last_repo = rid
                continue
        hint = f" (last repo tried: {last_repo!r} error: {last_exc})" if (last_repo and last_exc) else ""
        raise RuntimeError(
            f"WAN22: unable to resolve flow_shift from scheduler_config.json for spec={self.name!r}{hint}. "
            "Ensure vendored HF assets exist under apps/backend/huggingface/Wan-AI/**."
        ) from last_exc
    
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
    
    @property
    def t5_min_length(self) -> int:
        """T5 minimum length, always from FamilyRuntimeSpec."""
        return self._get_family_spec().t5_min_length or 512
    
    def is_14b(self) -> bool:
        return "14b" in self.name.lower()


def _k_predictor(spec: WanEngineSpec) -> FlowMatchEulerPrediction:
    """Create flow-match prediction for WAN."""
    logger.debug("Using FlowMatch predictor for WAN %s (shift=%.2f)", spec.name, spec.flow_shift)
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
    
    # VAE
    vae_model = codex_components.get("vae")
    if vae_model is None:
        raise ValueError("WAN runtime requires 'vae' component")
    vae = VAE(model=vae_model, family=ModelFamily.WAN22)
    
    # Transformer -> DenoiserPatcher
    transformer = codex_components.get("transformer")
    if transformer is None:
        raise ValueError("WAN runtime requires 'transformer' component")
    
    k_predictor = _k_predictor(spec)
    denoiser = DenoiserPatcher.from_model(
        model=transformer,
        diffusers_scheduler=None,
        k_predictor=k_predictor,
        config=estimated_config,
    )
    
    # T5 text encoder
    t5_encoder = codex_components.get("text_encoder")
    t5_tokenizer = codex_components.get("tokenizer")
    if t5_encoder is None or t5_tokenizer is None:
        raise ValueError("WAN runtime requires 'text_encoder' and 'tokenizer' components")
    
    t5_engine = T5TextProcessingEngine(
        text_encoder=t5_encoder,
        tokenizer=t5_tokenizer,
        emphasis_name=emphasis_name,
        min_length=spec.t5_min_length,
    )
    
    logger.info(
        "WAN runtime assembled: spec=%s device=%s dtype=%s",
        spec.name, device, dtype,
    )
    
    return WanEngineRuntime(
        vae=vae,
        denoiser=denoiser,
        text=WanTextPipelines(t5_text=t5_engine),
        device=device,
        dtype=dtype,
    )


# Pre-defined specs (using overrides only when different from FamilyRuntimeSpec)
WAN_14B_SPEC = WanEngineSpec(
    name="wan22_14b",
    # Uses defaults from FamilyRuntimeSpec for steps/cfg and resolves flow_shift from the vendored scheduler_config.json.
)

WAN_5B_SPEC = WanEngineSpec(
    name="wan22_5b",
    # 5B uses different defaults for steps/cfg; flow_shift is still resolved from scheduler_config.json.
    _default_steps_override=16,
    _default_cfg_override=6.0,
)

__all__ = [
    "WanTextPipelines",
    "WanEngineRuntime",
    "WanEngineSpec",
    "assemble_wan_runtime",
    "WAN_14B_SPEC",
    "WAN_5B_SPEC",
]
