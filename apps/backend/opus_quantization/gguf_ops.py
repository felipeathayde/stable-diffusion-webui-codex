import torch
import logging
import threading
from .compat import (
    ParameterGGUF as _BaseParameterGGUF,
    map_ggml_to_opus,
    _OpusQuantBridge,
    GGMLQuantizationType,
)
from .core import QuantType

# Simple, optional CPU LRU cache for dequantized GGUF tensors.
# Default: disabled. Can be enabled by set_cache_policy('cpu_lru', limit_mb).

_CACHE_LOCK = threading.Lock()
_CACHE_POLICY: str = 'none'  # 'none' | 'cpu_lru'
_CACHE_LIMIT_MB: int = 0
_CACHE_CUR_MB: int = 0
_CACHE: dict[int, torch.Tensor] = {}
_CACHE_ORDER: list[int] = []
_LOG = logging.getLogger("opus_quantization.gguf_ops")
if not _LOG.handlers:
    _LOG.addHandler(logging.StreamHandler())
_LOG.setLevel(logging.INFO)
_LOG.propagate = False
_HITS = 0
_MISSES = 0


def set_cache_policy(policy: str = 'none', limit_mb: int = 0) -> None:
    global _CACHE_POLICY, _CACHE_LIMIT_MB
    policy = (policy or 'none').strip().lower()
    with _CACHE_LOCK:
        _CACHE_POLICY = policy if policy in ('none', 'cpu_lru') else 'none'
        _CACHE_LIMIT_MB = int(max(0, limit_mb or 0))
        if _CACHE_POLICY == 'none' or _CACHE_LIMIT_MB == 0:
            _clear_cache_unlocked()
        _LOG.info("[gguf.ops] cache_policy=%s limit_mb=%d", _CACHE_POLICY, _CACHE_LIMIT_MB)


def _clear_cache_unlocked() -> None:
    global _CACHE, _CACHE_ORDER, _CACHE_CUR_MB
    _CACHE.clear()
    _CACHE_ORDER.clear()
    _CACHE_CUR_MB = 0


def clear_cache() -> None:
    with _CACHE_LOCK:
        _clear_cache_unlocked()
    _LOG.info("[gguf.ops] cache cleared")


def _tensor_size_mb(t: torch.Tensor) -> int:
    try:
        return int((t.nelement() * t.element_size()) / (1024 * 1024))
    except Exception:
        return 0


def _cache_get(tid: int) -> torch.Tensor | None:
    if _CACHE_POLICY != 'cpu_lru' or _CACHE_LIMIT_MB <= 0:
        return None
    with _CACHE_LOCK:
        t = _CACHE.get(tid)
        if t is not None:
            # Move to MRU position
            try:
                _CACHE_ORDER.remove(tid)
            except ValueError:
                pass
            _CACHE_ORDER.append(tid)
            global _HITS
            _HITS += 1
            if _HITS % 100 == 1:
                _LOG.info("[gguf.ops] cache hit: total_hits=%d size=%dMB items=%d", _HITS, _CACHE_CUR_MB, len(_CACHE_ORDER))
        return t


def _cache_put(tid: int, t: torch.Tensor) -> None:
    if _CACHE_POLICY != 'cpu_lru' or _CACHE_LIMIT_MB <= 0:
        return
    # We store CPU float tensors only
    if t.device.type != 'cpu':
        try:
            t = t.cpu()
        except Exception:
            return
    size_mb = _tensor_size_mb(t)
    if size_mb <= 0:
        return
    with _CACHE_LOCK:
        # Evict until enough room
        global _CACHE_CUR_MB
        while _CACHE_CUR_MB + size_mb > _CACHE_LIMIT_MB and _CACHE_ORDER:
            evict_id = _CACHE_ORDER.pop(0)
            ev = _CACHE.pop(evict_id, None)
            if ev is not None:
                _CACHE_CUR_MB -= max(0, _tensor_size_mb(ev))
        _CACHE[tid] = t
        _CACHE_ORDER.append(tid)
        _CACHE_CUR_MB += size_mb
        global _MISSES
        _MISSES += 1
        if _MISSES % 100 == 1:
            _LOG.info("[gguf.ops] cache store: total_misses=%d size=%dMB items=%d", _MISSES, _CACHE_CUR_MB, len(_CACHE_ORDER))


# OpusQuantization mapping - replaces old quants_mapping
# Now uses OpusQuantization kernels via the bridge
def _get_opus_bridge(ggml_type):
    """Get an OpusQuantBridge for a GGML type."""
    qtype = map_ggml_to_opus(ggml_type)
    if qtype is None:
        return None
    return _OpusQuantBridge(qtype)

# Legacy mapping for code that still uses quants_mapping directly
# Maps to OpusQuantBridge instances instead of old gguf.Q*_* classes
quants_mapping = {
    GGMLQuantizationType.Q2_K: _OpusQuantBridge(QuantType.Q2_K),
    GGMLQuantizationType.Q3_K: _OpusQuantBridge(QuantType.Q3_K),
    GGMLQuantizationType.Q4_0: _OpusQuantBridge(QuantType.Q4_0),
    GGMLQuantizationType.Q4_K: _OpusQuantBridge(QuantType.Q4_K),
    GGMLQuantizationType.Q4_1: _OpusQuantBridge(QuantType.Q4_1),
    GGMLQuantizationType.Q5_0: _OpusQuantBridge(QuantType.Q5_0),
    GGMLQuantizationType.Q5_1: _OpusQuantBridge(QuantType.Q5_1),
    GGMLQuantizationType.Q5_K: _OpusQuantBridge(QuantType.Q5_K),
    GGMLQuantizationType.Q6_K: _OpusQuantBridge(QuantType.Q6_K),
    GGMLQuantizationType.Q8_0: _OpusQuantBridge(QuantType.Q8_0),
    GGMLQuantizationType.BF16: _OpusQuantBridge(QuantType.BF16),
}


ParameterGGUF = _BaseParameterGGUF


def dequantize_tensor(tensor):
    if tensor is None:
        return None

    if not hasattr(tensor, 'gguf_cls'):
        return tensor

    gguf_cls = tensor.gguf_cls

    if gguf_cls is None:
        return tensor
    
    # Lazy bake: if tensor hasn't been baked yet, bake it now
    # This handles tensors that were created directly on CPU without going through .to()
    if hasattr(tensor, 'baked') and not tensor.baked:
        gguf_cls.bake(tensor)
    
    # Optional CPU LRU cache
    tid = id(tensor)
    cached = _cache_get(tid)
    if cached is not None:
        return cached
    out = gguf_cls.dequantize_pytorch(tensor)
    # Store CPU copy if caching is enabled
    try:
        _cache_put(tid, out)
    except Exception:
        pass
    return out
