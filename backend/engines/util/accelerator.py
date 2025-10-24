"""Shim module redirecting to apps.server.backend.engines.util.accelerator."""

from apps.server.backend.engines.util.accelerator import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith('_')]
