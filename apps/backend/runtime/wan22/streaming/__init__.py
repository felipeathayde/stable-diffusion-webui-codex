"""WAN streaming infrastructure using nn.Module pattern (format-agnostic).

This module provides block-based streaming for WanTransformer2DModel,
using the same approach as Flux streaming - operating on nn.Module
blocks rather than raw tensors.
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
