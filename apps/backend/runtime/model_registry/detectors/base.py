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
