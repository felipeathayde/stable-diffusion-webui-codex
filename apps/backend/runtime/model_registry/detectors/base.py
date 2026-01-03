"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Detector interface and registry for the model registry.
Defines the detector contract (`matches`/`build_signature`) and a global `REGISTRY` used by detector modules to self-register at import time.

Symbols (top-level; keep in sync; no ghosts):
- `ModelDetector` (protocol): Detector interface (priority + `matches()` + `build_signature()`).
- `DetectorRegistry` (class): Registry that stores detectors sorted by priority (lower runs first).
- `REGISTRY` (constant): Global registry instance used across detector modules.
"""

from __future__ import annotations

from typing import Protocol

from apps.backend.runtime.model_registry.signals import SignalBundle
from apps.backend.runtime.model_registry.specs import ModelSignature


class ModelDetector(Protocol):
    """Interface for detector implementations."""

    priority: int  # lower numbers run first

    def matches(self, bundle: SignalBundle) -> bool:
        """Return True when this detector recognizes the checkpoint."""

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:
        """Construct the structured signature for the checkpoint."""


class DetectorRegistry:
    def __init__(self) -> None:
        self._items: list[ModelDetector] = []

    def register(self, detector: ModelDetector) -> None:
        self._items.append(detector)
        self._items.sort(key=lambda d: getattr(d, "priority", 1000))

    @property
    def detectors(self) -> tuple[ModelDetector, ...]:
        return tuple(self._items)


REGISTRY = DetectorRegistry()
