"""Memory controller for WAN GGUF streaming with pluggable policies."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import torch

from apps.backend.runtime.ops.operations_gguf import dequantize_tensor

from .specs import WanBlockInfo

logger = logging.getLogger("backend.runtime.wan22.streaming.controller")


class WanStreamingPolicy(Enum):
    """Streaming policy for WAN GGUF tensors."""

    NAIVE = "naive"
    """Dequantize + load block tensors → forward → unload. Lowest VRAM, slowest."""

    WINDOW = "window"
    """Keep K most recent blocks' tensors on GPU, LRU eviction."""

    AGGRESSIVE = "aggressive"
    """Naive with prefetch of next block's tensors."""


@dataclass
class WanTransferStats:
    """Statistics for GGUF tensor transfers during streaming."""

    bytes_dequantized: int = 0
    bytes_to_gpu: int = 0
    bytes_to_cpu: int = 0
    transfers_to_gpu: int = 0
    transfers_to_cpu: int = 0
    total_time_ms: float = 0.0

    def record_to_gpu(self, bytes_count: int, time_ms: float) -> None:
        self.bytes_to_gpu += bytes_count
        self.transfers_to_gpu += 1
        self.total_time_ms += time_ms

    def record_to_cpu(self, bytes_count: int, time_ms: float) -> None:
        self.bytes_to_cpu += bytes_count
        self.transfers_to_cpu += 1
        self.total_time_ms += time_ms

    def record_dequantize(self, bytes_count: int) -> None:
        self.bytes_dequantized += bytes_count

    def summary(self) -> Dict[str, float]:
        return {
            "dequantized_mb": self.bytes_dequantized / (1024 * 1024),
            "to_gpu_mb": self.bytes_to_gpu / (1024 * 1024),
            "to_cpu_mb": self.bytes_to_cpu / (1024 * 1024),
            "transfers_to_gpu": self.transfers_to_gpu,
            "transfers_to_cpu": self.transfers_to_cpu,
            "total_time_ms": self.total_time_ms,
        }


@dataclass
class WanCoreController:
    """Memory controller managing GGUF tensor placement for WAN streaming.

    The controller tracks which block tensors are on GPU and handles
    dequantization + device transfer for GGUF quantized tensors.

    Attributes:
        storage_device: Device for offloaded tensors (typically CPU).
        compute_device: Device for active computation (typically CUDA).
        policy: Streaming policy determining eviction behavior.
        window_size: For WINDOW policy, number of blocks to keep on GPU.
        cache_dequantized: Whether to cache dequantized tensors.
    """

    storage_device: torch.device
    compute_device: torch.device
    policy: WanStreamingPolicy = WanStreamingPolicy.NAIVE
    window_size: int = 2
    cache_dequantized: bool = True

    # Cached dequantized tensors: key -> tensor
    _dequant_cache: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    # Which blocks have their tensors on GPU
    _on_gpu: Set[int] = field(default_factory=set, repr=False)
    # LRU tracking
    _access_order: List[int] = field(default_factory=list, repr=False)
    # Stats
    _stats: WanTransferStats = field(default_factory=WanTransferStats, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.storage_device, str):
            self.storage_device = torch.device(self.storage_device)
        if isinstance(self.compute_device, str):
            self.compute_device = torch.device(self.compute_device)

    @property
    def stats(self) -> WanTransferStats:
        return self._stats

    def reset(self) -> None:
        """Reset controller state between generations."""
        self._on_gpu.clear()
        self._access_order.clear()
        logger.debug("WAN controller state reset")

    def reset_stats(self) -> None:
        """Reset transfer statistics."""
        self._stats = WanTransferStats()

    def clear_cache(self) -> None:
        """Clear dequantized tensor cache."""
        self._dequant_cache.clear()
        logger.debug("Dequantization cache cleared")

    def is_on_gpu(self, block: WanBlockInfo) -> bool:
        """Check if block's tensors are on GPU."""
        return block.index in self._on_gpu

    def get_tensor(
        self,
        state: Dict[str, Any],
        key: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get a tensor, dequantizing GGUF if needed.

        Args:
            state: Model state dictionary.
            key: Tensor key to retrieve.
            dtype: Target dtype for the tensor.

        Returns:
            Dequantized tensor on compute device.
        """
        # Check cache first
        cache_key = f"{key}_{dtype}"
        if self.cache_dequantized and cache_key in self._dequant_cache:
            return self._dequant_cache[cache_key]

        tensor = state.get(key)
        if tensor is None:
            raise KeyError(f"Tensor not found in state: {key}")

        # Dequantize if GGUF
        if hasattr(tensor, "gguf_cls"):
            tensor = dequantize_tensor(tensor)
            self._stats.record_dequantize(tensor.numel() * tensor.element_size())

        # Move to compute device with target dtype
        tensor = tensor.to(device=self.compute_device, dtype=dtype)

        # Cache if enabled
        if self.cache_dequantized:
            self._dequant_cache[cache_key] = tensor

        return tensor

    def ensure_block_on_device(
        self,
        block: WanBlockInfo,
        state: Dict[str, Any],
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Ensure all block tensors are on compute device.

        Args:
            block: Block info with tensor keys.
            state: Model state dictionary.
            dtype: Target dtype for tensors.

        Returns:
            Dictionary of tensor key -> tensor on GPU.
        """
        if self.is_on_gpu(block):
            # Update LRU
            if block.index in self._access_order:
                self._access_order.remove(block.index)
            self._access_order.append(block.index)
            # Return cached tensors
            return {
                k: self._dequant_cache.get(f"{k}_{dtype}")
                for k in block.tensor_keys
                if f"{k}_{dtype}" in self._dequant_cache
            }

        start = time.perf_counter()
        tensors: Dict[str, torch.Tensor] = {}

        for key in block.tensor_keys:
            tensors[key] = self.get_tensor(state, key, dtype)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._on_gpu.add(block.index)
        self._access_order.append(block.index)
        self._stats.record_to_gpu(block.param_bytes, elapsed_ms)

        logger.debug(
            "Loaded block %d tensors to GPU (%d keys, %.2f MB, %.1f ms)",
            block.index,
            len(block.tensor_keys),
            block.param_bytes / (1024 * 1024),
            elapsed_ms,
        )

        return tensors

    def maybe_evict(self, block: WanBlockInfo) -> None:
        """Potentially evict block tensors based on policy.

        Args:
            block: Block that just finished executing.
        """
        if self.policy == WanStreamingPolicy.NAIVE:
            self._evict_block(block)
        elif self.policy == WanStreamingPolicy.WINDOW:
            if len(self._on_gpu) > self.window_size:
                self._evict_lru()
        elif self.policy == WanStreamingPolicy.AGGRESSIVE:
            self._evict_block(block)

    def _evict_block(self, block: WanBlockInfo) -> None:
        """Evict a specific block's tensors from cache."""
        if not self.is_on_gpu(block):
            return

        start = time.perf_counter()
        evicted_bytes = 0

        # Remove from dequant cache
        keys_to_remove = [
            k for k in self._dequant_cache.keys()
            if any(k.startswith(f"{tk}_") for tk in block.tensor_keys)
        ]
        for k in keys_to_remove:
            tensor = self._dequant_cache.pop(k, None)
            if tensor is not None:
                evicted_bytes += tensor.numel() * tensor.element_size()

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._on_gpu.discard(block.index)
        if block.index in self._access_order:
            self._access_order.remove(block.index)
        self._stats.record_to_cpu(evicted_bytes, elapsed_ms)

        logger.debug(
            "Evicted block %d tensors (%.2f MB, %.1f ms)",
            block.index,
            evicted_bytes / (1024 * 1024),
            elapsed_ms,
        )

    def _evict_lru(self) -> None:
        """Evict least recently used block."""
        if not self._access_order:
            return

        oldest_idx = self._access_order[0]
        # Find the block info (we only have index)
        self._on_gpu.discard(oldest_idx)
        self._access_order.pop(0)

        # Evict cached tensors for this block
        prefix = f"blocks.{oldest_idx}."
        keys_to_remove = [k for k in self._dequant_cache.keys() if prefix in k]
        for k in keys_to_remove:
            self._dequant_cache.pop(k, None)

        logger.debug("LRU evicted block %d", oldest_idx)

    def evict_all(self) -> None:
        """Evict all block tensors from GPU."""
        self._dequant_cache.clear()
        self._on_gpu.clear()
        self._access_order.clear()
        logger.debug("All WAN block tensors evicted")


def create_wan_controller(
    policy: str | WanStreamingPolicy = "naive",
    window_size: int = 2,
    storage_device: str = "cpu",
    compute_device: Optional[str] = None,
    cache_dequantized: bool = True,
) -> WanCoreController:
    """Factory function to create a WanCoreController.

    Args:
        policy: Streaming policy name or enum.
        window_size: Window size for WINDOW policy.
        storage_device: Offload device (default: "cpu").
        compute_device: Compute device (default: auto-detect CUDA).
        cache_dequantized: Whether to cache dequantized tensors.

    Returns:
        Configured WanCoreController instance.
    """
    if isinstance(policy, str):
        policy = WanStreamingPolicy(policy.lower())

    if compute_device is None:
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    return WanCoreController(
        storage_device=torch.device(storage_device),
        compute_device=torch.device(compute_device),
        policy=policy,
        window_size=window_size,
        cache_dequantized=cache_dequantized,
    )
