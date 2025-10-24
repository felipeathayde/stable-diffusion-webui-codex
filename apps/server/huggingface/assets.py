"""Shim redirecting to apps.server.backend.huggingface.assets."""

from apps.server.backend.huggingface.assets import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
