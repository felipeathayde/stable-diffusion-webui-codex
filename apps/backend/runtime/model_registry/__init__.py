"""Typed model registry for checkpoint detection (WIP)."""

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
