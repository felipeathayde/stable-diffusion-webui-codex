"""Helpers for stage-wise smart offload / fallback control."""

from __future__ import annotations

from typing import Dict


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


_SMART_CACHE_COUNTERS: Dict[str, Dict[str, int]] = {}


def _bucket(name: str) -> Dict[str, int]:
    bucket = _SMART_CACHE_COUNTERS.get(name)
    if bucket is None:
        bucket = {"hits": 0, "misses": 0}
        _SMART_CACHE_COUNTERS[name] = bucket
    return bucket


def record_smart_cache_hit(name: str) -> None:
    """Increment Smart Cache hit counter for the given bucket name."""
    try:
        bucket = _bucket(name)
        bucket["hits"] += 1
    except Exception:
        # Metrics must never interfere with runtime behaviour.
        pass


def record_smart_cache_miss(name: str) -> None:
    """Increment Smart Cache miss counter for the given bucket name."""
    try:
        bucket = _bucket(name)
        bucket["misses"] += 1
    except Exception:
        # Metrics must never interfere with runtime behaviour.
        pass


def get_smart_cache_stats() -> Dict[str, Dict[str, int]]:
    """Return a shallow copy of Smart Cache hit/miss counters."""
    return {name: dict(counts) for name, counts in _SMART_CACHE_COUNTERS.items()}


__all__ = [
    "smart_offload_enabled",
    "smart_fallback_enabled",
    "smart_cache_enabled",
    "record_smart_cache_hit",
    "record_smart_cache_miss",
    "get_smart_cache_stats",
]
