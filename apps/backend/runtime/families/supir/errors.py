"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR runtime error types.
Defines small, semantically distinct exception classes used by SUPIR validators and loaders so the API layer can fail loud with clear messages.

Symbols (top-level; keep in sync; no ghosts):
- `SupirError` (class): Base class for SUPIR runtime errors.
- `SupirConfigError` (class): Raised for invalid request configuration (bad types/ranges/unsupported combos).
- `SupirWeightsError` (class): Raised when SUPIR weights are missing/ambiguous/invalid.
- `SupirBaseModelError` (class): Raised when the selected SDXL base checkpoint is invalid (missing, non-SDXL, or SDXL refiner).
"""

from __future__ import annotations


class SupirError(RuntimeError):
    """Base class for SUPIR runtime errors."""


class SupirConfigError(SupirError):
    """Invalid SUPIR request configuration."""


class SupirWeightsError(SupirError):
    """SUPIR weights missing/invalid."""


class SupirBaseModelError(SupirError):
    """Invalid SUPIR base checkpoint selection."""


__all__ = [
    "SupirBaseModelError",
    "SupirConfigError",
    "SupirError",
    "SupirWeightsError",
]

