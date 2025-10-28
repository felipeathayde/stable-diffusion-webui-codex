"""Legacy shared-init shim removed.

Native bootstrap lives under ``apps.backend.codex.initialization``.
"""

from __future__ import annotations

raise RuntimeError(
    "modules.shared_init is no longer available. Use apps.backend.codex.initialization.initialize_codex"
    " for Codex backend bootstrap."
)

