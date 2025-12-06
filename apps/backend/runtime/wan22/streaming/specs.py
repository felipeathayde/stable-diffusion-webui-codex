"""Core dataclasses for WAN streaming infrastructure (nn.Module pattern).

This module mirrors the Flux streaming specs, operating on nn.Module
blocks rather than raw GGUF tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from apps.backend.runtime.wan22.model import WanTransformerBlock


@dataclass
class WanBlockInfo:
    """Metadata for a single WAN transformer block.

    Attributes:
        index: Block index in the model.
        module: Reference to the WanTransformerBlock nn.Module.
        param_bytes: Total parameter memory in bytes.
    """

    index: int
    module: nn.Module
    param_bytes: int = 0

    def __hash__(self) -> int:
        return hash(self.index)


@dataclass
class WanSegment:
    """A group of transformer blocks treated as a streaming unit.

    Segments are the atomic unit of CPU↔GPU transfer. Each segment
    contains one or more consecutive blocks.

    Attributes:
        name: Human-readable identifier (e.g., "wan_0_3").
        blocks: List of WanBlockInfo for blocks in this segment.
        param_bytes: Total parameter memory for all blocks in bytes.
    """

    name: str
    blocks: List[WanBlockInfo] = field(default_factory=list)
    param_bytes: int = 0

    @property
    def modules(self) -> List[nn.Module]:
        """Return list of nn.Module references for this segment."""
        return [b.module for b in self.blocks]

    def to_device(self, device: torch.device, *, non_blocking: bool = False) -> None:
        """Move all segment modules to the specified device."""
        for module in self.modules:
            module.to(device, non_blocking=non_blocking)

    def __len__(self) -> int:
        return len(self.blocks)


@dataclass
class WanExecutionPlan:
    """Ordered execution plan for streaming WAN transformer blocks.

    Attributes:
        segments: Ordered list of segments matching forward execution order.
        block_count: Total number of transformer blocks.
    """

    segments: List[WanSegment] = field(default_factory=list)
    block_count: int = 0

    @property
    def total_bytes(self) -> int:
        """Total parameter memory across all segments."""
        return sum(seg.param_bytes for seg in self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)


def calculate_module_bytes(module: nn.Module) -> int:
    """Calculate total parameter memory for a module in bytes."""
    total = 0
    for param in module.parameters():
        total += param.numel() * param.element_size()
    return total


def build_execution_plan(
    model: nn.Module,
    blocks_per_segment: int = 4,
) -> WanExecutionPlan:
    """Build an execution plan for streaming a WAN transformer.

    Args:
        model: WanTransformer2DModel with .blocks ModuleList.
        blocks_per_segment: Number of blocks to group per segment.

    Returns:
        WanExecutionPlan with ordered segments.
    """
    if not hasattr(model, "blocks"):
        raise ValueError("Model must have 'blocks' attribute (ModuleList)")

    blocks_list: List[WanBlockInfo] = []
    for idx, block in enumerate(model.blocks):
        info = WanBlockInfo(
            index=idx,
            module=block,
            param_bytes=calculate_module_bytes(block),
        )
        blocks_list.append(info)

    # Group into segments
    segments: List[WanSegment] = []
    for i in range(0, len(blocks_list), blocks_per_segment):
        chunk = blocks_list[i : i + blocks_per_segment]
        start_idx = chunk[0].index
        end_idx = chunk[-1].index
        name = f"wan_{start_idx}_{end_idx}"
        total_bytes = sum(b.param_bytes for b in chunk)

        segment = WanSegment(
            name=name,
            blocks=list(chunk),
            param_bytes=total_bytes,
        )
        segments.append(segment)

    return WanExecutionPlan(
        segments=segments,
        block_count=len(blocks_list),
    )
