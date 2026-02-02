"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Upscaler error types.
Defines explicit exceptions for discovery, load, and runtime failures so callers can fail loud with actionable messages.

Symbols (top-level; keep in sync; no ghosts):
- `UpscalerError` (exception): Base exception for upscaler failures.
- `UpscalerNotFoundError` (exception): Raised when an upscaler id cannot be resolved.
- `UpscalerLoadError` (exception): Raised when a model file cannot be loaded or is unsupported.
- `UpscalerRuntimeError` (exception): Raised when an upscaler fails during inference (including OOM when fallback is disabled).
"""

from __future__ import annotations


class UpscalerError(RuntimeError):
    """Base error for the upscalers runtime."""


class UpscalerNotFoundError(UpscalerError):
    """Raised when an upscaler id cannot be resolved."""


class UpscalerLoadError(UpscalerError):
    """Raised when an upscaler model cannot be loaded."""


class UpscalerRuntimeError(UpscalerError):
    """Raised when an upscaler fails during inference."""


__all__ = [
    "UpscalerError",
    "UpscalerNotFoundError",
    "UpscalerLoadError",
    "UpscalerRuntimeError",
]

