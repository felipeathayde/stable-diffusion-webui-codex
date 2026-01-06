"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed records for checkpoint and VAE assets discovered by the backend.
These are the canonical representations used by registries and public APIs, avoiding legacy-specific fields.

Symbols (top-level; keep in sync; no ghosts):
- `CheckpointFormat` (enum): Origin/layout type for checkpoint assets (single-file, diffusers folder, GGUF).
- `CheckpointPrediction` (enum): Prediction type hint for diffusion checkpoints (`epsilon`, `v_prediction`, `edm`, `unknown`).
- `CheckpointRecord` (dataclass): Metadata describing a discoverable checkpoint (paths, hashes, format, defaults, `as_dict()`).
- `VAERecord` (dataclass): Metadata describing a standalone VAE weights file (`as_dict()`).
- `__all__` (constant): Explicit export list for public record types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence


class CheckpointFormat(str, Enum):
    """Origin/structure of the checkpoint asset."""

    CHECKPOINT = "checkpoint"  # single-file .ckpt/.safetensors
    DIFFUSERS = "diffusers"    # folder with model_index + components
    GGUF = "gguf"              # quantized gguf weights (UNet etc.)


class CheckpointPrediction(str, Enum):
    UNKNOWN = "unknown"
    EPSILON = "epsilon"
    V_PREDICTION = "v_prediction"
    EDM = "edm"


@dataclass(frozen=True)
class CheckpointRecord:
    """Metadata describing a discoverable checkpoint."""

    name: str
    title: str
    filename: str
    path: str
    model_name: str
    format: CheckpointFormat
    sha256: str | None = None
    short_hash: str | None = None
    file_size: int | None = None
    default_dtype: str | None = None
    base_resolution: int | None = None
    prediction_type: CheckpointPrediction = CheckpointPrediction.UNKNOWN
    components: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)
    updated_at: float | None = None

    def as_dict(self) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {
            "name": self.name,
            "title": self.title,
            "filename": self.filename,
            "path": self.path,
            "model_name": self.model_name,
            "format": self.format.value,
        }
        if self.sha256:
            payload["sha256"] = self.sha256
        if self.short_hash:
            payload["short_hash"] = self.short_hash
        if self.file_size is not None:
            payload["file_size"] = self.file_size
        if self.default_dtype:
            payload["default_dtype"] = self.default_dtype
        if self.base_resolution is not None:
            payload["base_resolution"] = self.base_resolution
        if self.prediction_type is not CheckpointPrediction.UNKNOWN:
            payload["prediction_type"] = self.prediction_type.value
        if self.components:
            payload["components"] = list(self.components)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        return payload

    @property
    def directory(self) -> Path:
        return Path(self.path)


@dataclass(frozen=True)
class VAERecord:
    """Metadata for standalone VAE weights."""

    name: str
    filename: str
    source: str
    sha256: str | None = None
    short_hash: str | None = None
    updated_at: float | None = None

    def as_dict(self) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {
            "name": self.name,
            "filename": self.filename,
            "source": self.source,
        }
        if self.sha256:
            payload["sha256"] = self.sha256
        if self.short_hash:
            payload["short_hash"] = self.short_hash
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        return payload


__all__ = [
    "CheckpointFormat",
    "CheckpointPrediction",
    "CheckpointRecord",
    "VAERecord",
]
