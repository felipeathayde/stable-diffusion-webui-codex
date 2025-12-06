"""Memory controller for WAN streaming (nn.Module pattern).

This mirrors the Flux streaming controller, using .to(device) on
nn.Module blocks rather than GGUF-specific tensor operations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import torch

from .specs import WanSegment

logger = logging.getLogger("backend.runtime.wan22.streaming.controller")


class WanStreamingPolicy(Enum):
    """Streaming policy for WAN blocks."""

    NAIVE = "naive"
    """Load segment → forward → unload. Lowest VRAM, slowest."""

    WINDOW = "window"
    """Keep K most recent segments on GPU, LRU eviction."""

    AGGRESSIVE = "aggressive"
    """Naive with async prefetch of next segment."""


@dataclass
class TransferStats:
    """Statistics for CPU↔GPU transfers during streaming."""

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

    def summary(self) -> Dict[str, float]:
        return {
            "to_gpu_mb": self.bytes_to_gpu / (1024 * 1024),
            "to_cpu_mb": self.bytes_to_cpu / (1024 * 1024),
            "transfers_to_gpu": self.transfers_to_gpu,
            "transfers_to_cpu": self.transfers_to_cpu,
            "total_time_ms": self.total_time_ms,
        }


@dataclass
class WanCoreController:
    """Memory controller managing segment placement for WAN streaming.

    This is identical in design to Flux's CoreController, operating on
    nn.Module segments with .to(device) transfers.
    """

    storage_device: torch.device
    compute_device: torch.device
    policy: WanStreamingPolicy = WanStreamingPolicy.NAIVE
    window_size: int = 2
    non_blocking: bool = True

    _on_gpu: Set[str] = field(default_factory=set, repr=False)
    _access_order: List[str] = field(default_factory=list, repr=False)
    _stats: TransferStats = field(default_factory=TransferStats, repr=False)
    _prefetch_segment: Optional[WanSegment] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.storage_device, str):
            self.storage_device = torch.device(self.storage_device)
        if isinstance(self.compute_device, str):
            self.compute_device = torch.device(self.compute_device)

    @property
    def stats(self) -> TransferStats:
        return self._stats

    def reset(self) -> None:
        """Reset controller state between generations."""
        self._on_gpu.clear()
        self._access_order.clear()
        self._prefetch_segment = None

    def reset_stats(self) -> None:
        self._stats = TransferStats()

    def is_on_gpu(self, segment: WanSegment) -> bool:
        return segment.name in self._on_gpu

    def ensure_on_device(self, segment: WanSegment) -> None:
        """Ensure segment is on compute device (GPU)."""
        if self.is_on_gpu(segment):
            if segment.name in self._access_order:
                self._access_order.remove(segment.name)
            self._access_order.append(segment.name)
            return

        start = time.perf_counter()
        segment.to_device(self.compute_device, non_blocking=self.non_blocking)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._on_gpu.add(segment.name)
        self._access_order.append(segment.name)
        self._stats.record_to_gpu(segment.param_bytes, elapsed_ms)

        logger.debug(
            "Loaded segment '%s' to GPU (%.2f MB, %.1f ms)",
            segment.name,
            segment.param_bytes / (1024 * 1024),
            elapsed_ms,
        )

    def maybe_evict(self, segment: WanSegment, *, force: bool = False) -> None:
        """Potentially evict segment back to storage device."""
        if not self.is_on_gpu(segment) and not force:
            return

        if self.policy == WanStreamingPolicy.NAIVE or force:
            self._evict_segment(segment)
        elif self.policy == WanStreamingPolicy.WINDOW:
            if len(self._on_gpu) > self.window_size:
                self._evict_lru()
        elif self.policy == WanStreamingPolicy.AGGRESSIVE:
            self._evict_segment(segment)

    def _evict_segment(self, segment: WanSegment) -> None:
        if not self.is_on_gpu(segment):
            return

        start = time.perf_counter()
        segment.to_device(self.storage_device, non_blocking=self.non_blocking)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._on_gpu.discard(segment.name)
        if segment.name in self._access_order:
            self._access_order.remove(segment.name)
        self._stats.record_to_cpu(segment.param_bytes, elapsed_ms)

        logger.debug(
            "Evicted segment '%s' to CPU (%.2f MB, %.1f ms)",
            segment.name,
            segment.param_bytes / (1024 * 1024),
            elapsed_ms,
        )

    def _evict_lru(self) -> None:
        if not self._access_order:
            return

        for name in self._access_order:
            if name in self._on_gpu:
                self._on_gpu.discard(name)
                self._access_order.remove(name)
                logger.debug("LRU evicted segment '%s'", name)
                break

    def prefetch_next(self, next_segment: Optional[WanSegment]) -> None:
        """Hint to prefetch next segment (AGGRESSIVE policy)."""
        if self.policy != WanStreamingPolicy.AGGRESSIVE:
            return
        if next_segment is None or self.is_on_gpu(next_segment):
            return

        self._prefetch_segment = next_segment
        next_segment.to_device(self.compute_device, non_blocking=True)
        self._on_gpu.add(next_segment.name)
        logger.debug("Prefetching segment '%s'", next_segment.name)

    def evict_all(self) -> None:
        """Evict all segments from GPU."""
        self._on_gpu.clear()
        self._access_order.clear()
        self._prefetch_segment = None


def create_controller(
    policy: str | WanStreamingPolicy = "naive",
    window_size: int = 2,
    storage_device: str = "cpu",
    compute_device: Optional[str] = None,
) -> WanCoreController:
    """Factory function to create a WanCoreController."""
    if isinstance(policy, str):
        policy = WanStreamingPolicy(policy.lower())

    if compute_device is None:
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    return WanCoreController(
        storage_device=torch.device(storage_device),
        compute_device=torch.device(compute_device),
        policy=policy,
        window_size=window_size,
    )
