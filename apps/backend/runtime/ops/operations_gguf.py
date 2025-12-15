"""GGUF runtime operations (CodexQuantization-backed).

This module is the runtime integration layer for GGUF-quantized parameters:
- `CodexParameter` is the canonical packed-tensor type (no `gguf_cls` indirection).
- `dequantize_tensor()` returns float tensors for execution.
- Optional CPU LRU cache can store dequantized weights to reduce repeated work.
"""

from __future__ import annotations

import logging
import threading

import torch

from apps.backend.quantization.api import bake as codex_bake
from apps.backend.quantization.api import dequantize as codex_dequantize
from apps.backend.quantization.tensor import CodexParameter

__all__ = [
    "CodexParameter",
    "dequantize_tensor",
    "set_cache_policy",
    "clear_cache",
]

_LOG = logging.getLogger("backend.runtime.ops.operations_gguf")

_CACHE_LOCK = threading.Lock()
_CACHE_POLICY: str = "none"  # 'none' | 'cpu_lru'
_CACHE_LIMIT_MB: int = 0
_CACHE_CUR_MB: int = 0
_CACHE: dict[int, torch.Tensor] = {}
_CACHE_ORDER: list[int] = []


def set_cache_policy(policy: str = "none", limit_mb: int = 0) -> None:
    """Configure the optional dequant cache.

    - policy='cpu_lru': stores CPU float tensors only (never GPU tensors).
    """
    global _CACHE_POLICY, _CACHE_LIMIT_MB
    pol = (policy or "none").strip().lower()
    lim = int(max(0, limit_mb or 0))
    with _CACHE_LOCK:
        _CACHE_POLICY = pol if pol in ("none", "cpu_lru") else "none"
        _CACHE_LIMIT_MB = lim
        if _CACHE_POLICY == "none" or _CACHE_LIMIT_MB <= 0:
            _clear_cache_unlocked()
    _LOG.info("GGUF cache_policy=%s limit_mb=%d", _CACHE_POLICY, _CACHE_LIMIT_MB)


def clear_cache() -> None:
    with _CACHE_LOCK:
        _clear_cache_unlocked()
    _LOG.info("GGUF cache cleared")


def _clear_cache_unlocked() -> None:
    global _CACHE_CUR_MB
    _CACHE.clear()
    _CACHE_ORDER.clear()
    _CACHE_CUR_MB = 0


def _tensor_size_mb(t: torch.Tensor) -> int:
    try:
        return int((t.nelement() * t.element_size()) / (1024 * 1024))
    except Exception:
        return 0


def _cache_get(tid: int) -> torch.Tensor | None:
    if _CACHE_POLICY != "cpu_lru" or _CACHE_LIMIT_MB <= 0:
        return None
    with _CACHE_LOCK:
        t = _CACHE.get(tid)
        if t is None:
            return None
        try:
            _CACHE_ORDER.remove(tid)
        except ValueError:
            pass
        _CACHE_ORDER.append(tid)
        return t


def _cache_put(tid: int, t: torch.Tensor) -> None:
    if _CACHE_POLICY != "cpu_lru" or _CACHE_LIMIT_MB <= 0:
        return
    if t.device.type != "cpu":
        try:
            t = t.detach().cpu()
        except Exception:
            return
    size_mb = _tensor_size_mb(t)
    if size_mb <= 0:
        return
    with _CACHE_LOCK:
        global _CACHE_CUR_MB
        while _CACHE_CUR_MB + size_mb > _CACHE_LIMIT_MB and _CACHE_ORDER:
            evict_id = _CACHE_ORDER.pop(0)
            ev = _CACHE.pop(evict_id, None)
            if ev is not None:
                _CACHE_CUR_MB -= max(0, _tensor_size_mb(ev))
        _CACHE[tid] = t
        _CACHE_ORDER.append(tid)
        _CACHE_CUR_MB += size_mb


def dequantize_tensor(tensor):
    """Return a dequantized float tensor (or pass-through for non-quant tensors)."""
    if tensor is None:
        return None
    if not isinstance(tensor, CodexParameter) or tensor.qtype is None:
        return tensor

    if not tensor.baked:
        codex_bake(tensor)

    tid = id(tensor)
    cached = _cache_get(tid)
    if cached is not None:
        return cached

    out = codex_dequantize(tensor)
    _cache_put(tid, out)
    return out
