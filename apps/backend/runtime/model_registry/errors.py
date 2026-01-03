"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Error types for the typed model registry.
Provides explicit failures for detection (no matches / multiple matches) so callers can surface actionable diagnostics.

Symbols (top-level; keep in sync; no ghosts):
- `ModelRegistryError` (class): Base error for model registry failures.
- `UnknownModelError` (class): Raised when no detector matches a checkpoint.
- `AmbiguousModelError` (class): Raised when multiple detectors match the same checkpoint.
"""

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
