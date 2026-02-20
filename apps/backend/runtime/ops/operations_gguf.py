"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GGUF runtime operations backed by `apps.backend.quantization` (CodexQuantization).
Provides `dequantize_tensor(...)`, an optional CPU LRU cache for dequantized weights, and a run-scoped `dequant_forward` cache that can reuse
GGUF bake/dequant work across sampling steps (enabled explicitly; cleared at sampling cleanup, with disable INFO log emitted only when an active cache level was enabled).

Symbols (top-level; keep in sync; no ghosts):
- `CodexParameter` (class): Packed GGUF tensor wrapper (imported from `apps.backend.quantization.tensor`).
- `CodexPackLinearQ4KTilepackV1Parameter` (class): Packed CodexPack linear-weight wrapper (imported from `apps.backend.quantization.codexpack_tensor`).
- `set_cache_policy` (function): Configure optional CPU LRU dequant cache (`none` or `cpu_lru`) and size limit.
- `clear_cache` (function): Clear cached dequantized tensors.
- `enable_dequant_forward_cache` (function): Enable per-run GGUF cache (`lvl1` or `lvl2`) with a memory cap.
- `disable_dequant_forward_cache` (function): Disable per-run GGUF cache and free cached tensors.
- `is_dequant_forward_cache_enabled` (function): Returns True when per-run GGUF cache is active for this thread.
- `dequantize_tensor_for_forward` (function): Dequantize a GGUF tensor for forward, optionally using the per-run cache.
- `dequantize_tensor` (function): Dequantize a `CodexParameter` to a float tensor (pass-through for non-GGUF tensors).
- `__all__` (constant): Public export list for GGUF runtime operations.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import torch

from apps.backend.quantization.api import dequantize as codex_dequantize
from apps.backend.quantization.codexpack_tensor import CodexPackLinearQ4KTilepackV1Parameter
from apps.backend.quantization.tensor import CodexParameter

__all__ = [
    "CodexPackLinearQ4KTilepackV1Parameter",
    "CodexParameter",
    "dequantize_tensor_for_forward",
    "dequantize_tensor",
    "set_cache_policy",
    "clear_cache",
    "enable_dequant_forward_cache",
    "disable_dequant_forward_cache",
    "is_dequant_forward_cache_enabled",
]

_LOG = logging.getLogger("backend.runtime.ops.operations_gguf")

_CACHE_LOCK = threading.Lock()
_CACHE_POLICY: str = "none"  # 'none' | 'cpu_lru'
_CACHE_LIMIT_MB: int = 0
_CACHE_CUR_MB: int = 0
_CACHE: dict[int, torch.Tensor] = {}
_CACHE_ORDER: list[int] = []

_FORWARD_CACHE_LOCAL = threading.local()

_FORWARD_CACHE_ALLOWED_LEVELS = ("off", "lvl1", "lvl2")


@dataclass(slots=True)
class _ForwardDequantCache:
    level: str = "off"
    limit_bytes: int = 0
    used_bytes: int = 0
    used_bytes_moved: int = 0
    used_bytes_dequant: int = 0
    moved_params: dict[tuple[int, str], CodexParameter] = field(default_factory=dict)
    dequant_tensors: dict[tuple[int, str, torch.dtype], torch.Tensor] = field(default_factory=dict)
    calls: int = 0
    passthrough: int = 0
    moved_hits: int = 0
    moved_stores: int = 0
    moved_skips: int = 0
    dequant_hits: int = 0
    dequant_stores: int = 0
    dequant_skips: int = 0

    def clear(self) -> None:
        self.moved_params.clear()
        self.dequant_tensors.clear()
        self.used_bytes = 0
        self.used_bytes_moved = 0
        self.used_bytes_dequant = 0
        self.calls = 0
        self.passthrough = 0
        self.moved_hits = 0
        self.moved_stores = 0
        self.moved_skips = 0
        self.dequant_hits = 0
        self.dequant_stores = 0
        self.dequant_skips = 0


def enable_dequant_forward_cache(*, level: str, limit_mb: int) -> None:
    """Enable per-run GGUF cache for `dequant_forward` execution.

    - lvl1: cache moved+baked GGUF params per-run (reduces repeated bake/move churn).
    - lvl2: also cache dequantized float tensors per-run (bigger speed win; more memory).

    The cache is thread-local; callers must disable it at the end of the sampling run.
    """

    lvl = (level or "off").strip().lower()
    if lvl not in _FORWARD_CACHE_ALLOWED_LEVELS:
        allowed = ", ".join(_FORWARD_CACHE_ALLOWED_LEVELS)
        raise ValueError(f"GGUF dequant cache level must be one of: {allowed} (got {level!r}).")
    lim = int(limit_mb)
    if lvl != "off" and lim <= 0:
        raise ValueError("GGUF dequant cache requires limit_mb > 0.")

    cache = _ForwardDequantCache(level=lvl, limit_bytes=max(lim, 0) * 1024 * 1024)
    setattr(_FORWARD_CACHE_LOCAL, "cache", cache)
    _LOG.info("GGUF dequant_forward cache enabled (level=%s limit_mb=%d)", lvl, lim)
    if _LOG.isEnabledFor(logging.DEBUG):
        cuda_mem = _cuda_mem_mb()
        if cuda_mem is not None:
            free_mb, total_mb, allocated_mb, reserved_mb, device_id = cuda_mem
            _LOG.debug(
                "GGUF dequant_forward cache cuda mem (enable): device=%s free_mb=%d total_mb=%d "
                "allocated_mb=%d reserved_mb=%d",
                device_id,
                free_mb,
                total_mb,
                allocated_mb,
                reserved_mb,
            )


def disable_dequant_forward_cache() -> None:
    cache: _ForwardDequantCache | None = getattr(_FORWARD_CACHE_LOCAL, "cache", None)
    was_enabled = cache is not None and getattr(cache, "level", "off") in {"lvl1", "lvl2"}
    if cache is not None:
        if _LOG.isEnabledFor(logging.DEBUG):
            accounted_mb = int(max(0, cache.used_bytes) // (1024 * 1024))
            accounted_moved_mb = int(max(0, cache.used_bytes_moved) // (1024 * 1024))
            accounted_dequant_mb = int(max(0, cache.used_bytes_dequant) // (1024 * 1024))
            limit_mb = int(max(0, cache.limit_bytes) // (1024 * 1024))
            cuda_mem_before = _cuda_mem_mb()
            _LOG.debug(
                "GGUF dequant_forward cache stats: level=%s accounted_mb=%d/%d accounted_moved_mb=%d "
                "accounted_dequant_mb=%d moved_params=%d dequant_tensors=%d "
                "calls=%d passthrough=%d moved_hits=%d moved_stores=%d moved_skips=%d "
                "dequant_hits=%d dequant_stores=%d dequant_skips=%d",
                cache.level,
                accounted_mb,
                limit_mb,
                accounted_moved_mb,
                accounted_dequant_mb,
                len(cache.moved_params),
                len(cache.dequant_tensors),
                cache.calls,
                cache.passthrough,
                cache.moved_hits,
                cache.moved_stores,
                cache.moved_skips,
                cache.dequant_hits,
                cache.dequant_stores,
                cache.dequant_skips,
            )
            if cuda_mem_before is not None:
                free_mb, total_mb, allocated_mb, reserved_mb, device_id = cuda_mem_before
                _LOG.debug(
                    "GGUF dequant_forward cache cuda mem (before clear): device=%s free_mb=%d total_mb=%d "
                    "allocated_mb=%d reserved_mb=%d",
                    device_id,
                    free_mb,
                    total_mb,
                    allocated_mb,
                    reserved_mb,
                )
        cache.clear()
        if _LOG.isEnabledFor(logging.DEBUG):
            cuda_mem_after = _cuda_mem_mb()
            if cuda_mem_after is not None:
                free_mb, total_mb, allocated_mb, reserved_mb, device_id = cuda_mem_after
                _LOG.debug(
                    "GGUF dequant_forward cache cuda mem (after clear): device=%s free_mb=%d total_mb=%d "
                    "allocated_mb=%d reserved_mb=%d",
                    device_id,
                    free_mb,
                    total_mb,
                    allocated_mb,
                    reserved_mb,
                )
    setattr(_FORWARD_CACHE_LOCAL, "cache", None)
    if was_enabled:
        _LOG.info("GGUF dequant_forward cache disabled")


def is_dequant_forward_cache_enabled() -> bool:
    cache: _ForwardDequantCache | None = getattr(_FORWARD_CACHE_LOCAL, "cache", None)
    return cache is not None and cache.level in {"lvl1", "lvl2"} and cache.limit_bytes > 0


def _forward_cache_get() -> _ForwardDequantCache | None:
    cache: _ForwardDequantCache | None = getattr(_FORWARD_CACHE_LOCAL, "cache", None)
    if cache is None or cache.level not in {"lvl1", "lvl2"} or cache.limit_bytes <= 0:
        return None
    return cache


def _tensor_nbytes(t: torch.Tensor) -> int:
    try:
        return int(t.nelement() * t.element_size())
    except Exception:
        return 0


def _cuda_mem_mb() -> tuple[int, int, int, int, str] | None:
    """Best-effort CUDA memory snapshot in MB.

    Returns:
        (free_mb, total_mb, allocated_mb, reserved_mb, device_id)
    """

    if not torch.cuda.is_available():
        return None
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        allocated_b = int(torch.cuda.memory_allocated())
        reserved_b = int(torch.cuda.memory_reserved())
        mb = 1024 * 1024
        device_id = f"cuda:{torch.cuda.current_device()}"
        return (
            int(free_b // mb),
            int(total_b // mb),
            int(allocated_b // mb),
            int(reserved_b // mb),
            device_id,
        )
    except Exception:
        return None


def _forward_cache_can_store(cache: _ForwardDequantCache, *, nbytes: int) -> bool:
    if nbytes <= 0:
        return False
    if nbytes > cache.limit_bytes:
        return False
    if cache.used_bytes + nbytes > cache.limit_bytes:
        return False
    return True


def _forward_cache_account(cache: _ForwardDequantCache, *, nbytes: int, bucket: str) -> None:
    n = max(0, int(nbytes))
    cache.used_bytes += n
    if bucket == "moved":
        cache.used_bytes_moved += n
    elif bucket == "dequant":
        cache.used_bytes_dequant += n
    else:
        raise ValueError(f"Unknown forward cache accounting bucket: {bucket!r}")


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

    # CPU cache stores CPU float tensors only; don't accidentally turn GPU execution
    # into CPU->GPU transfers by returning cached CPU weights for GPU-resident params.
    use_cache = tensor.device.type == "cpu"

    tid = id(tensor)
    cached = _cache_get(tid) if use_cache else None
    if cached is not None:
        return cached

    out = codex_dequantize(tensor)
    if use_cache:
        _cache_put(tid, out)
    return out


def dequantize_tensor_for_forward(
    tensor: torch.Tensor | None,
    *,
    target_device: torch.device,
    target_dtype: torch.dtype | None,
    non_blocking: bool = False,
) -> torch.Tensor | None:
    """Dequantize a GGUF tensor for forward, optionally using the per-run cache.

    This is designed to be used by `operations.get_weight_and_bias(...)` when the cache scope is enabled.
    """

    cache = _forward_cache_get()
    if cache is None:
        if tensor is None:
            return None
        if not isinstance(tensor, CodexParameter) or tensor.qtype is None:
            return tensor
        moved = tensor.to(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
        return codex_dequantize(moved)

    cache.calls += 1
    if tensor is None:
        return None
    if not isinstance(tensor, CodexParameter) or tensor.qtype is None:
        cache.passthrough += 1
        return tensor

    source_id = id(tensor)
    device_key = str(target_device)
    moved_key = (source_id, device_key)
    moved = cache.moved_params.get(moved_key)
    if moved is None:
        moved_candidate = tensor.to(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
        moved_bytes = 0 if moved_candidate is tensor else _tensor_nbytes(moved_candidate.data)
        if moved_bytes == 0:
            cache.moved_params[moved_key] = moved_candidate
            cache.moved_stores += 1
            moved = moved_candidate
        elif _forward_cache_can_store(cache, nbytes=moved_bytes):
            cache.moved_params[moved_key] = moved_candidate
            _forward_cache_account(cache, nbytes=moved_bytes, bucket="moved")
            cache.moved_stores += 1
            moved = moved_candidate
        else:
            cache.moved_skips += 1
            moved = moved_candidate
    else:
        cache.moved_hits += 1
        if target_dtype is not None and getattr(moved, "computation_dtype", None) != target_dtype:
            moved = moved.to(dtype=target_dtype)

    if cache.level != "lvl2":
        return codex_dequantize(moved)

    dtype_key = target_dtype or getattr(moved, "computation_dtype", torch.float32)
    out_key = (source_id, device_key, dtype_key)
    cached = cache.dequant_tensors.get(out_key)
    if cached is not None:
        cache.dequant_hits += 1
        return cached

    out = codex_dequantize(moved)
    out_bytes = _tensor_nbytes(out)
    if _forward_cache_can_store(cache, nbytes=out_bytes):
        cache.dequant_tensors[out_key] = out
        _forward_cache_account(cache, nbytes=out_bytes, bucket="dequant")
        cache.dequant_stores += 1
    else:
        cache.dequant_skips += 1
    return out
