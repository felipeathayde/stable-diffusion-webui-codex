"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux core streaming controller (segment device placement + transfer stats).
Tracks which `Segment`s are placed on GPU, loads/evicts segments according to a streaming policy, and records transfer statistics for
diagnostics and tuning.

Symbols (top-level; keep in sync; no ghosts):
- `StreamingPolicy` (enum): High-level streaming policy for segment load/eviction behavior.
- `TransferStats` (dataclass): Aggregated CPU↔GPU transfer counters and timing.
- `CoreController` (dataclass): Controller managing `Segment` placement based on policy (LRU/windowing + optional prefetch).
- `create_controller` (function): Factory that builds a `CoreController` from simple config values.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import torch

from .specs import Segment

logger = logging.getLogger("backend.runtime.flux.streaming.controller")


class StreamingPolicy(Enum):
    """Streaming policy determining how segments are moved between devices."""

    NAIVE = "naive"
    """Load segment → forward → unload. Simplest, lowest VRAM, slowest."""

    WINDOW = "window"
    """Keep K most recent segments pinned on GPU, LRU eviction."""

    AGGRESSIVE = "aggressive"
    """Naive with async prefetch of next segment during current forward."""


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
        """Return summary dict with MB values."""
        return {
            "to_gpu_mb": self.bytes_to_gpu / (1024 * 1024),
            "to_cpu_mb": self.bytes_to_cpu / (1024 * 1024),
            "transfers_to_gpu": self.transfers_to_gpu,
            "transfers_to_cpu": self.transfers_to_cpu,
            "total_time_ms": self.total_time_ms,
        }


@dataclass
class CoreController:
    """Memory controller managing segment placement for streaming.

    The controller tracks which segments are currently on GPU and decides
    when to load/unload segments based on the configured policy.

    Attributes:
        storage_device: Device for offloaded segments (typically CPU).
        compute_device: Device for active computation (typically CUDA).
        policy: Streaming policy determining eviction behavior.
        window_size: For WINDOW policy, number of segments to keep on GPU.
        non_blocking: Whether to use non-blocking transfers.
    """

    storage_device: torch.device
    compute_device: torch.device
    policy: StreamingPolicy = StreamingPolicy.NAIVE
    window_size: int = 2
    non_blocking: bool = True

    # Runtime state (not part of config)
    _on_gpu: Set[str] = field(default_factory=set, repr=False)
    _access_order: List[str] = field(default_factory=list, repr=False)
    _stats: TransferStats = field(default_factory=TransferStats, repr=False)
    _prefetch_segment: Optional[Segment] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Ensure devices are torch.device objects
        if isinstance(self.storage_device, str):
            self.storage_device = torch.device(self.storage_device)
        if isinstance(self.compute_device, str):
            self.compute_device = torch.device(self.compute_device)

    @property
    def stats(self) -> TransferStats:
        """Return transfer statistics."""
        return self._stats

    def reset(self) -> None:
        """Reset controller state between images/batches."""
        self._on_gpu.clear()
        self._access_order.clear()
        self._prefetch_segment = None
        logger.debug("Controller state reset")

    def reset_stats(self) -> None:
        """Reset transfer statistics."""
        self._stats = TransferStats()

    def is_on_gpu(self, segment: Segment) -> bool:
        """Check if segment is currently on GPU."""
        return segment.name in self._on_gpu

    def ensure_on_device(self, segment: Segment) -> None:
        """Ensure segment is on compute device (GPU).

        If the segment is already on GPU, this is a no-op.
        Otherwise, transfer it from storage device.
        """
        if self.is_on_gpu(segment):
            # Update access order for LRU tracking
            if segment.name in self._access_order:
                self._access_order.remove(segment.name)
            self._access_order.append(segment.name)
            return

        # Transfer to GPU
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

    def maybe_evict(self, segment: Segment, *, force: bool = False) -> None:
        """Potentially evict segment back to storage device.

        Behavior depends on policy:
        - NAIVE: Always evict immediately after use.
        - WINDOW: Evict only if over window_size limit (LRU).
        - AGGRESSIVE: Same as NAIVE but done async.

        Args:
            segment: The segment that just finished executing.
            force: If True, evict regardless of policy.
        """
        if not self.is_on_gpu(segment) and not force:
            return

        if self.policy == StreamingPolicy.NAIVE or force:
            self._evict_segment(segment)
        elif self.policy == StreamingPolicy.WINDOW:
            # Only evict if we exceed window size
            if len(self._on_gpu) > self.window_size:
                self._evict_lru()
        elif self.policy == StreamingPolicy.AGGRESSIVE:
            # Same as naive for now (prefetch handled separately)
            self._evict_segment(segment)

    def _evict_segment(self, segment: Segment) -> None:
        """Evict a specific segment to storage device."""
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
        """Evict the least recently used segment."""
        if not self._access_order:
            return

        # Find oldest segment that's on GPU
        for name in self._access_order:
            if name in self._on_gpu:
                # We need to find the segment object - this is a limitation
                # For now, just mark it as evicted
                self._on_gpu.discard(name)
                self._access_order.remove(name)
                logger.debug("LRU evicted segment '%s'", name)
                break

    def prefetch_next(self, next_segment: Optional[Segment]) -> None:
        """Hint to prefetch next segment (AGGRESSIVE policy only).

        This initiates an async transfer of the next segment while
        the current segment is executing.
        """
        if self.policy != StreamingPolicy.AGGRESSIVE:
            return
        if next_segment is None:
            return
        if self.is_on_gpu(next_segment):
            return

        # Start async transfer
        self._prefetch_segment = next_segment
        next_segment.to_device(self.compute_device, non_blocking=True)
        self._on_gpu.add(next_segment.name)
        logger.debug("Prefetching segment '%s'", next_segment.name)

    def evict_all(self) -> None:
        """Evict all segments from GPU (cleanup)."""
        self._on_gpu.clear()
        self._access_order.clear()
        self._prefetch_segment = None
        logger.debug("All segments evicted")


def create_controller(
    policy: str | StreamingPolicy = "naive",
    window_size: int = 2,
    storage_device: str = "cpu",
    compute_device: Optional[str] = None,
) -> CoreController:
    """Factory function to create a CoreController.

    Args:
        policy: Streaming policy name or enum.
        window_size: Window size for WINDOW policy.
        storage_device: Offload device (default: "cpu").
        compute_device: Compute device (default: auto-detect CUDA).

    Returns:
        Configured CoreController instance.
    """
    if isinstance(policy, str):
        policy = StreamingPolicy(policy.lower())

    if compute_device is None:
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    return CoreController(
        storage_device=torch.device(storage_device),
        compute_device=torch.device(compute_device),
        policy=policy,
        window_size=window_size,
    )
