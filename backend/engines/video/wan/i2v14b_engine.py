"""Shim module redirecting to apps.server.backend.engines.video.wan.i2v14b_engine."""

from apps.server.backend.engines.video.wan.i2v14b_engine import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith('_')]
