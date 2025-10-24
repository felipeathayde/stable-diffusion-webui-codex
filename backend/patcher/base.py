"""Shim module redirecting to apps.server.backend.patchers.base."""

from apps.server.backend.patchers.base import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
