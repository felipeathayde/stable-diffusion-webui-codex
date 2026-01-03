"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Exception hierarchy for Codex-native CLIP vision runtime modules.
Separates config/load/input errors so callers can surface actionable failure messages without silent fallbacks.

Symbols (top-level; keep in sync; no ghosts):
- `ClipVisionError` (class): Base exception for CLIP vision runtime issues.
- `ClipVisionConfigError` (class): Raised when a configuration specification is invalid or unsupported.
- `ClipVisionLoadError` (class): Raised when loading a checkpoint/state dict fails.
- `ClipVisionInputError` (class): Raised when caller-provided tensors are malformed or incompatible.
"""

from __future__ import annotations


class ClipVisionError(RuntimeError):
    """Base exception for clip vision runtime issues."""


class ClipVisionConfigError(ClipVisionError):
    """Raised when a configuration specification is invalid or unsupported."""


class ClipVisionLoadError(ClipVisionError):
    """Raised when loading a checkpoint/state dict fails."""


class ClipVisionInputError(ClipVisionError):
    """Raised when caller-provided tensors are malformed or incompatible."""
