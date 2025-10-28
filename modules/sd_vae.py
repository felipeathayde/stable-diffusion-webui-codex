"""Legacy VAE registry removed."""

from __future__ import annotations

raise RuntimeError(
    "modules.sd_vae has been removed. Use apps.backend.runtime.models.api.list_vaes to discover VAE weights "
    "or configure VAEs via Codex backend options."
)

