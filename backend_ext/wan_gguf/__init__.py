"""Shim redirecting to apps.server.backend.wan_gguf."""

from apps.server.backend.wan_gguf import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith('_')]
