"""Legacy shim for exception types.

Canonical module lives in `apps.server.backend.core.exceptions`.
"""

from apps.server.backend.core.exceptions import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith('_')]
