"""Shim module redirecting to apps.server.backend.core.requests."""

from apps.server.backend.core.requests import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith('_')]
