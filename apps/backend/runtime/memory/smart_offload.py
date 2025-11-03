"""Helpers for stage-wise smart offload control."""

from __future__ import annotations

import os


def _env_enabled() -> bool:
    value = os.getenv("CODEX_SMART_OFFLOAD")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def smart_offload_enabled() -> bool:
    """Return True when smart offload is enabled via CLI or environment."""

    if _env_enabled():
        return True

    try:
        from apps.backend.infra.config import args as cfg

        namespace = getattr(cfg, "args", None)
        return bool(getattr(namespace, "smart_offload", False))
    except Exception:
        return False


__all__ = ["smart_offload_enabled"]
