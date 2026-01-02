"""Helpers for stage-wise smart offload / fallback control."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Dict, Iterator


def _snapshot():
    from apps.backend.codex import options as codex_options

    return codex_options.get_snapshot()


_THREAD_OVERRIDES = threading.local()


def _get_override(name: str) -> bool | None:
    return getattr(_THREAD_OVERRIDES, name, None)


@contextmanager
def smart_runtime_overrides(
    *,
    smart_offload: bool | None = None,
    smart_fallback: bool | None = None,
    smart_cache: bool | None = None,
) -> Iterator[None]:
    """Temporarily override smart flags for the current thread.

    This is intended for per-request overrides inside worker threads, so runtime
    code that consults these helpers (sampling, memory manager, patchers) can
    honor request-level flags without relying on persisted `/api/options`.
    """
    prev_offload = _get_override("smart_offload")
    prev_fallback = _get_override("smart_fallback")
    prev_cache = _get_override("smart_cache")
    _THREAD_OVERRIDES.smart_offload = smart_offload
    _THREAD_OVERRIDES.smart_fallback = smart_fallback
    _THREAD_OVERRIDES.smart_cache = smart_cache
    try:
        yield
    finally:
        _THREAD_OVERRIDES.smart_offload = prev_offload
        _THREAD_OVERRIDES.smart_fallback = prev_fallback
        _THREAD_OVERRIDES.smart_cache = prev_cache


def smart_offload_enabled() -> bool:
    """Return True when smart offload is enabled (Codex options only)."""
    override = _get_override("smart_offload")
    if override is not None:
        return bool(override)
    try:
        snap = _snapshot()
        return bool(getattr(snap, "codex_smart_offload", False))
    except Exception:
        return False


def smart_fallback_enabled() -> bool:
    """Return True when smart CPU fallback on OOM is enabled (Codex options only)."""
    override = _get_override("smart_fallback")
    if override is not None:
        return bool(override)
    try:
        snap = _snapshot()
        return bool(getattr(snap, "codex_smart_fallback", False))
    except Exception:
        return False


def smart_cache_enabled() -> bool:
    """Return True when SDXL smart caching (TE + embed_values) is enabled."""
    override = _get_override("smart_cache")
    if override is not None:
        return bool(override)
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
    "smart_runtime_overrides",
    "record_smart_cache_hit",
    "record_smart_cache_miss",
    "get_smart_cache_stats",
]
