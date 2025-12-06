"""Flux core streaming infrastructure for reduced VRAM usage.

This module provides block-based streaming for FluxTransformer2DModel,
enabling inference on lower-VRAM GPUs by loading/unloading transformer
block segments dynamically during the denoising loop.

Usage:
    from apps.backend.runtime.flux.streaming import (
        trace_execution_plan,
        CoreController,
        StreamedFluxCore,
        StreamingPolicy,
    )
    
    plan = trace_execution_plan(flux_core)
    controller = CoreController(storage="cpu", compute="cuda", policy=StreamingPolicy.NAIVE)
    streamed = StreamedFluxCore(flux_core, plan, controller)
"""

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

