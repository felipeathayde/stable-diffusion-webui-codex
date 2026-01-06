"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public API for Flux core streaming (execution-plan tracing, controller policy, and wrapper).

Symbols (top-level; keep in sync; no ghosts):
- `Segment` (dataclass): Streaming segment grouping consecutive transformer blocks.
- `ExecutionPlan` (dataclass): Ordered segment execution plan for a Flux transformer core.
- `BlockType` (enum): Block type classification for streaming (double vs single).
- `trace_execution_plan` (function): Trace a Flux transformer core into an `ExecutionPlan`.
- `CoreController` (class): Streaming controller that moves segments between devices per policy.
- `StreamingPolicy` (enum): Streaming policy selection for `CoreController`.
- `StreamedFluxCore` (class): Wrapper that streams a Flux core using an execution plan + controller.
- `StreamingConfig` (dataclass): User-facing config for enabling/tuning core streaming.
"""

# Usage (sketch):
#   from apps.backend.runtime.flux.streaming import trace_execution_plan, CoreController, StreamedFluxCore, StreamingPolicy
#   plan = trace_execution_plan(flux_core)
#   controller = CoreController(storage="cpu", compute="cuda", policy=StreamingPolicy.NAIVE)
#   streamed = StreamedFluxCore(flux_core, plan, controller)

from .specs import Segment, ExecutionPlan, BlockType
from .tracer import trace_execution_plan
from .controller import CoreController, StreamingPolicy
from .wrapper import StreamedFluxCore
from .config import StreamingConfig

__all__ = [
    "Segment",
    "ExecutionPlan",
    "BlockType",
    "trace_execution_plan",
    "CoreController",
    "StreamingPolicy",
    "StreamedFluxCore",
    "StreamingConfig",
]
