"""Custom exceptions for Codex clip vision runtime."""

from __future__ import annotations


class ClipVisionError(RuntimeError):
    """Base exception for clip vision runtime issues."""


class ClipVisionConfigError(ClipVisionError):
    """Raised when a configuration specification is invalid or unsupported."""


class ClipVisionLoadError(ClipVisionError):
    """Raised when loading a checkpoint/state dict fails."""


class ClipVisionInputError(ClipVisionError):
    """Raised when caller-provided tensors are malformed or incompatible."""
