"""Shim module redirecting to apps.server.backend.config.args."""

from apps.server.backend.config.args import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
