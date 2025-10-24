"""Shim redirecting to apps.server.backend.huggingface."""

from apps.server.backend.huggingface import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
