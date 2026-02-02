"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR VAE precompute cache (default OFF).
Provides a small, in-process LRU cache intended to store SUPIR Stage-1 precompute tensors on CPU
when the caller enables `supir_use_vae_cache`.

This cache is deliberately small and explicit:
- no silent enablement,
- no unbounded growth,
- a `clear()` escape hatch.

Symbols (top-level; keep in sync; no ghosts):
- `SupirVaeCache` (class): Small thread-safe LRU cache.
- `get_supir_vae_cache` (function): Returns the process-global cache instance.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class _CacheStats:
    max_items: int
    size: int
    hits: int
    misses: int


class SupirVaeCache:
    """Thread-safe LRU cache for CPU-resident Stage-1 precompute artifacts."""

    def __init__(self, *, max_items: int = 1) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be > 0")
        self._max_items = int(max_items)
        self._lock = threading.Lock()
        self._items: "OrderedDict[str, object]" = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[object]:
        k = str(key or "").strip()
        if not k:
            return None
        with self._lock:
            v = self._items.get(k)
            if v is None:
                self._misses += 1
                return None
            self._hits += 1
            self._items.move_to_end(k, last=True)
            return v

    def put(self, key: str, value: object) -> None:
        k = str(key or "").strip()
        if not k:
            raise ValueError("cache key must be non-empty")
        with self._lock:
            self._items[k] = value
            self._items.move_to_end(k, last=True)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            s = _CacheStats(
                max_items=self._max_items,
                size=len(self._items),
                hits=self._hits,
                misses=self._misses,
            )
        return {
            "max_items": s.max_items,
            "size": s.size,
            "hits": s.hits,
            "misses": s.misses,
        }


_CACHE: SupirVaeCache | None = None
_CACHE_LOCK = threading.Lock()


def get_supir_vae_cache(*, max_items: int = 1) -> SupirVaeCache:
    """Return the process-global SUPIR VAE cache instance."""

    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None or _CACHE._max_items != int(max_items):
            _CACHE = SupirVaeCache(max_items=max_items)
        return _CACHE


__all__ = ["SupirVaeCache", "get_supir_vae_cache"]

