"""Shim module redirecting to apps.server.backend.runtime.attention."""

from apps.server.backend.runtime.attention import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
