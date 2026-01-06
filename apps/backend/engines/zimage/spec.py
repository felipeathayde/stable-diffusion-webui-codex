"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z Image engine specification and runtime assembly (analogous to Flux `spec.py`).
Builds a `ZImageEngineRuntime` from parsed components, optionally loading external VAE/text-encoder assets when the checkpoint is core-only.

Symbols (top-level; keep in sync; no ghosts):
- `ZImageCLIP` (class): CLIP-like wrapper that exposes a `ModelPatcher` for memory management integration around the Z-Image text encoder.
- `ZImageTextPipelines` (dataclass): Text processing pipeline bundle for the Z Image engine (currently Qwen3 text).
- `ZImageEngineRuntime` (dataclass): Runtime container for Z Image engine components (VAE/denoiser/text pipelines/clip wrapper/device/dtype).
- `ZImageEngineSpec` (dataclass): Engine specification delegating defaults to `FamilyRuntimeSpec` with optional overrides.
- `_k_predictor` (function): Builds the flow-match predictor used by the denoiser patcher for Z Image sampling.
- `_load_external_vae` (function): Loads an external VAE asset for core-only checkpoints.
- `_load_external_text_encoder` (function): Loads an external Qwen3 text encoder asset for core-only checkpoints.
- `assemble_zimage_runtime` (function): Assembles the runtime (including external assets when required) and returns a `ZImageEngineRuntime`.
- `ZIMAGE_SPEC` (constant): Default Z Image engine spec instance.
- `__all__` (constant): Public export list for this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from apps.backend.patchers.base import ModelPatcher
from apps.backend.patchers.denoiser import DenoiserPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.model_registry.family_runtime import get_family_spec, FamilyRuntimeSpec
from apps.backend.runtime.modules.k_prediction import FlowMatchEulerPrediction
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole

logger = logging.getLogger("backend.engines.zimage.spec")


class ZImageCLIP:
    """CLIP-like wrapper for Z Image text encoder with memory management support.
    
    This wrapper provides a `patcher` attribute that integrates with the
    memory management system, allowing automatic GPU loading/offloading.
    """
    
    def __init__(self, text_encoder):
        """Initialize with a ZImageTextEncoder.
        
        Args:
            text_encoder: A ZImageTextEncoder instance with a `model` attribute.
        """
        self.text_encoder = text_encoder
        
        # Create ModelPatcher for memory management integration
        load_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
        offload_device = memory_management.manager.get_offload_device(DeviceRole.TEXT_ENCODER)
        
        # The patcher wraps the underlying model, enabling load_model_gpu/unload_model
        self.patcher = ModelPatcher(
            text_encoder.model,
            load_device=load_device,
            offload_device=offload_device,
        )


@dataclass(frozen=True)
class ZImageTextPipelines:
    """Text processing pipeline for Z Image."""
    qwen3_text: Any  # ZImageTextProcessingEngine


@dataclass
class ZImageEngineRuntime:
    """Runtime container for Z Image engine components."""
    vae: VAE
    denoiser: DenoiserPatcher  # wraps ZImageTransformer2DModel
    text: ZImageTextPipelines
    clip: ZImageCLIP  # wrapper with ModelPatcher for memory management
    device: str = "cuda"
    dtype: str = "bf16"


@dataclass(frozen=True)
class ZImageEngineSpec:
    """Specification for Z Image engine.
    
    This spec delegates to FamilyRuntimeSpec for default values,
    with optional per-variant overrides.
    """
    name: str = "zimage"
    family: ModelFamily = ModelFamily.ZIMAGE
    
    # Optional overrides (if None, delegates to FamilyRuntimeSpec)
    _flow_shift_override: Optional[float] = field(default=None, repr=False)
    _default_steps_override: Optional[int] = field(default=None, repr=False)
    _default_cfg_override: Optional[float] = field(default=None, repr=False)
    
    def _get_family_spec(self) -> FamilyRuntimeSpec:
        """Get the FamilyRuntimeSpec for this engine."""
        return get_family_spec(self.family)
    
    @property
    def flow_shift(self) -> float:
        """Flow-match shift, delegating to FamilyRuntimeSpec if not overridden.
        
        Z Image Turbo uses shift=3.0 (matches HF `scheduler_config.json` for Z-Image-Turbo).
        """
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
    # Turbo uses linear schedule (shift=1.0); standard models use shift=3.0
    return FlowMatchEulerPrediction(pseudo_timestep_range=1000, mu=spec.flow_shift)


def _load_external_vae(vae_path: str | None, dtype: str = "bf16") -> object:
    """Load VAE from external path for core-only checkpoints.
    
    Uses the shared Flow16 VAE loader since Z Image uses the same
    16-channel latent space as Flux.
    """
    import torch
    from apps.backend.runtime.common.vae import load_flow16_vae
    
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16 if dtype == "fp16" else torch.float32
    
    if vae_path is None or not str(vae_path).strip():
        raise ValueError(
            "Z Image core-only checkpoint requires an external VAE. "
            "Please select a VAE so the request can include 'vae_sha' (sha256)."
        )
    
    return load_flow16_vae(vae_path, dtype=torch_dtype)


def _load_external_text_encoder(tenc_path: str | None, dtype: str = "bf16") -> object:
    """Load Qwen3 text encoder from external path for core-only checkpoints."""
    import torch
    
    # Text encoder path is required - no automatic fallback
    if tenc_path is None:
        raise ValueError(
            "Z Image core-only checkpoint requires external text encoder (Qwen3-4B). "
            "Please select one in the UI or place it in models/zimage-tenc/"
        )
    
    logger.info("Loading external text encoder from: %s", tenc_path)
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16 if dtype == "fp16" else torch.float32
    
    # Detect if GGUF or safetensors
    if tenc_path.lower().endswith(".gguf"):
        # Load GGUF text encoder
        from apps.backend.runtime.zimage.text_encoder import ZImageTextEncoder
        encoder = ZImageTextEncoder.from_gguf(tenc_path, torch_dtype=torch_dtype)
    else:
        # Load safetensors text encoder
        from safetensors.torch import load_file
        from apps.backend.runtime.zimage.text_encoder import ZImageTextEncoder
        state_dict = load_file(tenc_path)
        encoder = ZImageTextEncoder.from_state_dict(state_dict, torch_dtype=torch_dtype)
    
    return encoder


def assemble_zimage_runtime(
    *,
    spec: ZImageEngineSpec,
    codex_components: Mapping[str, object],
    estimated_config: Any,
    device: str = "cuda",
    dtype: str = "bf16",
    vae_path: str | None = None,
    tenc_path: str | None = None,
) -> ZImageEngineRuntime:
    """Assemble Z Image runtime from components.
    
    Args:
        spec: Engine specification.
        codex_components: Dict with 'transformer', optionally 'vae', 'text_encoder'.
        estimated_config: Model config.
        device: Target device.
        dtype: Target dtype.
        vae_path: Optional external VAE path (required when the checkpoint is core-only).
        tenc_path: Optional external text encoder path (required when the checkpoint is core-only).
    
    Returns:
        Assembled ZImageEngineRuntime.
    """
    import torch
    
    def _log_vram(label: str) -> None:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            alloc = torch.cuda.memory_allocated()
            logger.info("[zimage-assemble] %s: free=%.2f GB, alloc=%.2f GB", label, free/1e9, alloc/1e9)
    
    logger.debug("Assembling Z Image runtime")
    _log_vram("START")
    
    # Get transformer (always required)
    transformer = codex_components.get("transformer")
    if transformer is None:
        raise ValueError("Z Image requires 'transformer' component")
    
    _log_vram("AFTER get transformer")
    
    # Detect core-only checkpoints (no VAE or text encoder embedded in components).
    base_vae = codex_components.get("vae")
    base_text_encoder = codex_components.get("text_encoder")
    is_core_only = base_vae is None or base_text_encoder is None

    vae_model = base_vae
    text_encoder = base_text_encoder

    external_vae = str(vae_path or "").strip() or None
    external_tenc = str(tenc_path or "").strip() or None

    # External assets are opt-in for full checkpoints, and required for core-only
    # checkpoints (GGUF / transformer-only exports).
    if external_vae is not None:
        logger.info("Z Image: loading external VAE (vae_path=%s)", external_vae)
        vae_model = _load_external_vae(external_vae, dtype=dtype)
        _log_vram("AFTER load external VAE")
    if vae_model is None:
        if is_core_only:
            _load_external_vae(None, dtype=dtype)  # raises with an actionable message
        raise ValueError("Z Image checkpoint did not include a VAE; provide an external VAE via 'vae_sha'.")

    if external_tenc is not None:
        logger.info("Z Image: loading external text encoder (tenc_path=%s)", external_tenc)
        text_encoder = _load_external_text_encoder(external_tenc, dtype=dtype)
        _log_vram("AFTER load external TEnc")
    if text_encoder is None:
        if is_core_only:
            _load_external_text_encoder(None, dtype=dtype)  # raises with an actionable message
        raise ValueError("Z Image checkpoint did not include a text encoder; provide one via 'tenc_sha'.")
    
    # Wrap VAE
    vae = VAE(model=vae_model, family=ModelFamily.ZIMAGE)
    _log_vram("AFTER VAE wrapper")
    
    # Wrap transformer in DenoiserPatcher
    k_predictor = _k_predictor(spec)
    denoiser = DenoiserPatcher.from_model(
        model=transformer,
        diffusers_scheduler=None,
        k_predictor=k_predictor,
        config=estimated_config,
    )
    _log_vram("AFTER DenoiserPatcher.from_model")
    
    # Wrap text encoder with ZImageCLIP for memory management
    clip = ZImageCLIP(text_encoder)
    _log_vram("AFTER ZImageCLIP wrapper")
    
    # Create text processing engine
    from apps.backend.runtime.zimage.text_encoder import ZImageTextProcessingEngine
    text_engine = ZImageTextProcessingEngine(text_encoder)
    
    _log_vram("FINAL")
    logger.info("Z Image runtime assembled: device=%s dtype=%s core_only=%s", device, dtype, is_core_only)
    
    return ZImageEngineRuntime(
        vae=vae,
        denoiser=denoiser,
        text=ZImageTextPipelines(qwen3_text=text_engine),
        clip=clip,
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
