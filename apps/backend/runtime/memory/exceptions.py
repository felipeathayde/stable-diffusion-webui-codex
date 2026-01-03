"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Custom exception types raised by the Codex memory manager/config stack.

Symbols (top-level; keep in sync; no ghosts):
- `MemoryConfigurationError` (class): Raised when the runtime memory config is invalid or unsupported.
- `MemoryLoadError` (class): Raised when models cannot be loaded/unloaded according to policy.
- `HardwareProbeError` (class): Raised when hardware probing fails or returns inconsistent data.
"""

from __future__ import annotations


class MemoryConfigurationError(RuntimeError):
    """Raised when the runtime memory configuration is invalid or unsupported."""


class MemoryLoadError(RuntimeError):
    """Raised when models cannot be loaded/unloaded according to policy."""


class HardwareProbeError(RuntimeError):
    """Raised when hardware probing fails or returns inconsistent data."""


__all__ = ["HardwareProbeError", "MemoryConfigurationError", "MemoryLoadError"]
