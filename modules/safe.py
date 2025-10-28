"""Legacy checkpoint safety helpers removed."""

from __future__ import annotations

raise RuntimeError(
    "modules.safe has been removed. Use apps.backend.runtime.models.safety for checkpoint validation "
    "and loading."
)

