"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public facade for Codex-native CLIP vision runtime helpers.
Exposes the main `ClipVisionEncoder` wrapper plus core error and output types.

Symbols (top-level; keep in sync; no ghosts):
- `ClipVisionEncoder` (class): Runtime wrapper for CLIP vision encoders (load + encode with device/dtype management).
- `ClipVisionError` (class): Base exception for CLIP vision runtime failures.
- `ClipVisionConfigError` (class): Raised when a vision spec/config is invalid or unsupported.
- `ClipVisionInputError` (class): Raised when caller-provided tensors/inputs are malformed.
- `ClipVisionLoadError` (class): Raised when loading a vision checkpoint/state dict fails.
- `ClipVisionOutput` (class): Structured encoder output bundle (hidden states + embeddings).
"""

from .encoder import ClipVisionEncoder
from .errors import (
    ClipVisionConfigError,
    ClipVisionError,
    ClipVisionInputError,
    ClipVisionLoadError,
)
from .types import ClipVisionOutput

__all__ = [
    "ClipVisionEncoder",
    "ClipVisionError",
    "ClipVisionConfigError",
    "ClipVisionInputError",
    "ClipVisionLoadError",
    "ClipVisionOutput",
]
