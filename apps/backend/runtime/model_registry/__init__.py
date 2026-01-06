"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public re-export surface for the typed model registry.
Exposes the state-dict detection entrypoint and the shared signature/spec types used across detectors and runtime assembly.

Symbols (top-level; keep in sync; no ghosts):
- `detect_from_state_dict` (function): Runs detectors over a state dict and returns a `ModelSignature` (raises `UnknownModelError`/`AmbiguousModelError`).
- `ModelFamily` (enum): High-level checkpoint family tags (SD15/SDXL/Flux/WAN22/etc).
- `PredictionKind` (enum): Model prediction parameterization tags (`eps`, `v_prediction`, `flow`, etc).
- `LatentFormat` (enum): Latent space format tags used for runtime assembly.
- `QuantizationKind` (enum): Quantization scheme identifiers (`none`, `nf4`, `fp4`, `gguf`).
- `QuantizationHint` (dataclass): Structured quantization hint (kind + optional detail).
- `CodexCoreArchitecture` (enum): Core network architecture identifiers (UNet/DiT/Transformer/FlowTransformer).
- `CodexCoreSignature` (dataclass): Core network signature (channels, context dim, depth, key prefixes).
- `TextEncoderSignature` (dataclass): Text encoder signature metadata (name/prefix/expected dim/tokenizer hint).
- `VAESignature` (dataclass): VAE signature metadata (key prefix + latent channels).
- `ModelSignature` (dataclass): Structured signature for a checkpoint (produced by detectors).
- `FamilyRuntimeSpec` (dataclass): Runtime defaults/invariants for a `ModelFamily` (latent/conditioning + sampling defaults).
- `FAMILY_RUNTIME_SPECS` (constant): Mapping of `ModelFamily` to `FamilyRuntimeSpec`.
- `get_family_spec` (function): Returns runtime spec for a known family (raises on missing).
- `get_family_spec_or_default` (function): Returns runtime spec for a family, falling back to an explicit default.
- `ModelRegistryError` (class): Base error for model registry failures.
- `UnknownModelError` (class): Raised when no detector matches a checkpoint.
- `AmbiguousModelError` (class): Raised when multiple detectors match the same checkpoint.
"""

from __future__ import annotations

from .loader import detect_from_state_dict
from .specs import (
    ModelFamily,
    PredictionKind,
    LatentFormat,
    QuantizationKind,
    QuantizationHint,
    CodexCoreArchitecture,
    CodexCoreSignature,
    TextEncoderSignature,
    VAESignature,
    ModelSignature,
)
from .family_runtime import (
    FamilyRuntimeSpec,
    FAMILY_RUNTIME_SPECS,
    get_family_spec,
    get_family_spec_or_default,
)
from .errors import ModelRegistryError, UnknownModelError, AmbiguousModelError

__all__ = [
    "detect_from_state_dict",
    "ModelFamily",
    "PredictionKind",
    "LatentFormat",
    "QuantizationKind",
    "QuantizationHint",
    "CodexCoreArchitecture",
    "CodexCoreSignature",
    "TextEncoderSignature",
    "VAESignature",
    "ModelSignature",
    "FamilyRuntimeSpec",
    "FAMILY_RUNTIME_SPECS",
    "get_family_spec",
    "get_family_spec_or_default",
    "ModelRegistryError",
    "UnknownModelError",
    "AmbiguousModelError",
]
