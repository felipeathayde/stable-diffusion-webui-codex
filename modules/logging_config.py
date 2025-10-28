"""Legacy logging configuration shim removed.

Use ``apps.backend.runtime.logging.setup_logging`` for native Codex logging.
"""

from __future__ import annotations

raise RuntimeError(
    "modules.logging_config has been removed. Call apps.backend.runtime.logging.setup_logging"
    " to configure logging for Codex backend components."
)

