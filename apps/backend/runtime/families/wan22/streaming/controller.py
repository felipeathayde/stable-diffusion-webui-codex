"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Memory controller for WAN22 core streaming (shared core wrapper).
WAN22 and Flux share the same controller implementation to keep streaming semantics identical and avoid drift. This module keeps the WAN22
public import path stable while delegating the implementation to `apps.backend.runtime.streaming.controller`.

Symbols (top-level; keep in sync; no ghosts):
- `WanStreamingPolicy` (enum): Streaming policy (`naive`/`window`/`aggressive`) controlling segment residency.
- `TransferStats` (dataclass): Tracks CPU↔GPU transfer bytes/counts/time for streaming telemetry.
- `WanCoreController` (class): Core streaming controller; loads/unloads segments around forward passes and manages LRU/window policies.
- `create_controller` (function): Factory that builds a `WanCoreController` from policy and blocks-per-segment inputs.
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

from .specs import WanSegment

logger = logging.getLogger("backend.runtime.wan22.streaming.controller")

WanStreamingPolicy = StreamingPolicy


class WanCoreController(_StreamingController[WanSegment]):
    """WAN22 streaming controller (shared implementation)."""


def create_controller(
    policy: str | WanStreamingPolicy = "naive",
    window_size: int = 2,
    storage_device: str = "cpu",
    compute_device: Optional[str] = None,
) -> WanCoreController:
    if isinstance(policy, str):
        policy = WanStreamingPolicy(policy.lower())

    if compute_device is None:
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    return WanCoreController(
        storage_device=torch.device(storage_device),
        compute_device=torch.device(compute_device),
        policy=policy,
        window_size=window_size,
        non_blocking=True,
        logger=logger,
    )

