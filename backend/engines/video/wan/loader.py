"""Shim module redirecting to apps.server.backend.engines.video.wan.loader."""

from apps.server.backend.engines.video.wan.loader import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith('_')]
