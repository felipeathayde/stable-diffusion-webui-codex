"""Shim module redirecting to apps.server.backend.video.interpolation."""

from apps.server.backend.video.interpolation import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith('_')]
