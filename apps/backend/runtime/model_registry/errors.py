from __future__ import annotations

class ModelRegistryError(RuntimeError):
    """Base error for model registry failures."""


class UnknownModelError(ModelRegistryError):
    """Raised when no detector matches a given checkpoint."""

    def __init__(self, message: str, *, detail: dict | None = None):
        super().__init__(message)
        self.detail = detail or {}


class AmbiguousModelError(ModelRegistryError):
    """Raised when multiple detectors match the same checkpoint."""

    def __init__(self, message: str, *, matches: list[str]):
        super().__init__(message)
        self.matches = matches
