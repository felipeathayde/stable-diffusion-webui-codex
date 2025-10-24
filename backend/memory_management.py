"""Shim module redirecting to apps.server.backend.runtime.memory."""

from apps.server.backend.runtime.memory.memory_management import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
