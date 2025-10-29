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
    "ModelRegistryError",
    "UnknownModelError",
    "AmbiguousModelError",
]
