"""Legacy SD model type annotations removed."""

from __future__ import annotations

raise RuntimeError(
    "modules.sd_models_types has been removed. The Codex backend exposes structured checkpoint data via "
    "apps.backend.runtime.models.types."
)

