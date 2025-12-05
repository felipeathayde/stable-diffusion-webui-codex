"""Helpers for stage-wise smart offload / fallback control."""

from __future__ import annotations


def _snapshot():
    from apps.backend.codex import options as codex_options

    return codex_options.get_snapshot()


def smart_offload_enabled() -> bool:
    """Return True when smart offload is enabled (Codex options only)."""
    try:
        snap = _snapshot()
        return bool(getattr(snap, "codex_smart_offload", False))
    except Exception:
        return False


def smart_fallback_enabled() -> bool:
    """Return True when smart CPU fallback on OOM is enabled (Codex options only)."""
    try:
        snap = _snapshot()
        return bool(getattr(snap, "codex_smart_fallback", False))
    except Exception:
        return False


def smart_cache_enabled() -> bool:
    """Return True when SDXL smart caching (TE + embed_values) is enabled."""
    try:
        snap = _snapshot()
        return bool(getattr(snap, "codex_smart_cache", False))
    except Exception:
        return False


__all__ = ["smart_offload_enabled", "smart_fallback_enabled", "smart_cache_enabled"]
