"""Shim module redirecting to apps.server.backend.engines.util.attention_backend."""

from apps.server.backend.engines.util.attention_backend import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith('_')]
