"""Custom exception types used by the Codex memory manager."""

from __future__ import annotations


class MemoryConfigurationError(RuntimeError):
    """Raised when the runtime memory configuration is invalid or unsupported."""


class MemoryLoadError(RuntimeError):
    """Raised when models cannot be loaded/unloaded according to policy."""


class HardwareProbeError(RuntimeError):
    """Raised when hardware probing fails or returns inconsistent data."""


__all__ = ["HardwareProbeError", "MemoryConfigurationError", "MemoryLoadError"]
