"""WAN 2.2 core streaming infrastructure for reduced VRAM usage.

This module provides block-based streaming for WanDiTGGUF models,
enabling video generation on lower-VRAM GPUs by loading/unloading
GGUF tensors per-block during the denoising loop.

Unlike Flux (nn.Module-based), WAN uses GGUF tensors stored in a
state dict, requiring tensor-level streaming rather than module-level.
"""

from .specs import WanBlockInfo, WanExecutionPlan
from .controller import WanCoreController, WanStreamingPolicy
from .wrapper import StreamedWanDiTGGUF

__all__ = [
    "WanBlockInfo",
    "WanExecutionPlan",
    "WanCoreController",
    "WanStreamingPolicy",
    "StreamedWanDiTGGUF",
]
