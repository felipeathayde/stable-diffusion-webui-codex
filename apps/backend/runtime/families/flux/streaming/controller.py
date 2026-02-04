"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux core streaming controller (shared core wrapper).
Flux and WAN22 share the same controller implementation to keep streaming semantics identical and avoid drift. This module keeps the
Flux-family public import path stable while delegating the implementation to `apps.backend.runtime.streaming.controller`.

Symbols (top-level; keep in sync; no ghosts):
- `StreamingPolicy` (enum): High-level streaming policy for segment load/eviction behavior.
- `TransferStats` (dataclass): Aggregated CPU↔GPU transfer counters and timing.
- `CoreController` (dataclass): Controller managing `Segment` placement based on policy (LRU/windowing + optional prefetch).
- `create_controller` (function): Factory that builds a `CoreController` from simple config values.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from apps.backend.runtime.streaming.controller import (
    StreamingController as _StreamingController,
    StreamingPolicy,
    TransferStats,
)

from .specs import Segment

logger = logging.getLogger("backend.runtime.flux.streaming.controller")


class CoreController(_StreamingController[Segment]):
    """Flux streaming controller (shared implementation)."""


def create_controller(
    policy: str | StreamingPolicy = "naive",
    window_size: int = 2,
    storage_device: str = "cpu",
    compute_device: Optional[str] = None,
) -> CoreController:
    if isinstance(policy, str):
        policy = StreamingPolicy(policy.lower())

    if compute_device is None:
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    return CoreController(
        storage_device=torch.device(storage_device),
        compute_device=torch.device(compute_device),
        policy=policy,
        window_size=window_size,
        non_blocking=True,
        logger=logger,
    )

