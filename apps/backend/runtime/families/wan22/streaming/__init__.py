"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN core streaming facade (nn.Module-based; format-agnostic).
Re-exports the execution-plan specs, controller/policy, and wrapper used to stream `WanTransformer2DModel` blocks.

Symbols (top-level; keep in sync; no ghosts):
- `WanBlockInfo` (class): Metadata for a single WAN transformer block (index/module/bytes).
- `WanSegment` (class): Streaming segment containing one or more consecutive blocks.
- `WanExecutionPlan` (class): Ordered plan describing the forward execution of WAN segments.
- `build_execution_plan` (function): Builds an execution plan by grouping model blocks into segments.
- `WanCoreController` (class): Core streaming controller coordinating CPU↔GPU transfers.
- `WanStreamingPolicy` (class): Streaming policy enum used by the controller.
- `StreamedWanTransformer` (class): Wrapper that executes a WAN core using a streaming plan/controller.
"""

from .specs import WanBlockInfo, WanSegment, WanExecutionPlan, build_execution_plan
from .controller import WanCoreController, WanStreamingPolicy
from .wrapper import StreamedWanTransformer

__all__ = [
    "WanBlockInfo",
    "WanSegment",
    "WanExecutionPlan",
    "build_execution_plan",
    "WanCoreController",
    "WanStreamingPolicy",
    "StreamedWanTransformer",
]
