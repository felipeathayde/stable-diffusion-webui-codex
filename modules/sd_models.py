"""Legacy checkpoint registry removed."""

from __future__ import annotations

raise RuntimeError(
    "modules.sd_models has been removed. Query checkpoints via apps.backend.runtime.models.api or "
    "load engines through apps.backend.codex.loader.load_engine."
)

