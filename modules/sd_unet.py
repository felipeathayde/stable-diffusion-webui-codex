"""Legacy UNet override registry removed."""

from __future__ import annotations

raise RuntimeError(
    "modules.sd_unet has been removed. Implement UNet customisations against the Codex backend runtime or "
    "engine patchers explicitly."
)

