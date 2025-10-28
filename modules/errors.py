"""Legacy error helpers removed.

This module previously proxied error reporting for the legacy WebUI stack.
Codex backend now exposes native helpers in ``apps.backend.runtime.errors``;
importing this module is therefore unsupported.
"""

from __future__ import annotations

raise RuntimeError(
    "modules.errors has been removed. Import apps.backend.runtime.errors and use its"
    " ErrorRegistry/report_error helpers instead."
)

