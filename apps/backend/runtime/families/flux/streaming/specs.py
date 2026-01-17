"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Dataclasses/enums describing Flux streaming blocks, segments, and execution plans.

Symbols (top-level; keep in sync; no ghosts):
- `BlockType` (enum): Flux transformer block type classification (double vs single stream).
- `BlockInfo` (dataclass): Metadata about a single transformer block (index/type/module/bytes).
- `Segment` (dataclass): Streaming unit grouping consecutive blocks of the same type.
- `ExecutionPlan` (dataclass): Ordered plan describing segment execution order and total sizes.
- `calculate_module_bytes` (function): Sum parameter bytes for a module (for VRAM accounting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    pass


class BlockType(Enum):
    """Type of transformer block in Flux architecture."""

    DOUBLE = "double"  # DoubleStreamBlock (dual-stream img+txt)
    SINGLE = "single"  # SingleStreamBlock (concatenated tokens)


@dataclass
class BlockInfo:
    """Metadata for a single transformer block.

    Attributes:
        index: Original index in the parent ModuleList.
        block_type: Whether this is a double or single stream block.
        module: Reference to the actual nn.Module.
        param_bytes: Total parameter memory in bytes.
    """

    index: int
    block_type: BlockType
    module: nn.Module
    param_bytes: int

    def __hash__(self) -> int:
        return hash((self.block_type, self.index))


@dataclass
class Segment:
    """A group of transformer blocks treated as a streaming unit.

    Segments are the atomic unit of CPU↔GPU transfer. Each segment
    contains one or more consecutive blocks of the same type.

    Attributes:
        name: Human-readable identifier (e.g., "double_0_3", "single_4_7").
        blocks: List of BlockInfo for blocks in this segment.
        param_bytes: Total parameter memory for all blocks in bytes.
    """

    name: str
    blocks: List[BlockInfo] = field(default_factory=list)
    param_bytes: int = 0

    @property
    def modules(self) -> List[nn.Module]:
        """Return list of nn.Module references for this segment."""
        return [b.module for b in self.blocks]

    @property
    def block_type(self) -> BlockType:
        """Return the block type (all blocks in a segment share the same type)."""
        if not self.blocks:
            raise ValueError("Segment has no blocks")
        return self.blocks[0].block_type

    def to_device(self, device: torch.device, *, non_blocking: bool = False) -> None:
        """Move all segment modules to the specified device."""
        for module in self.modules:
            module.to(device, non_blocking=non_blocking)

    def __len__(self) -> int:
        return len(self.blocks)


@dataclass
class ExecutionPlan:
    """Ordered execution plan for streaming Flux transformer blocks.

    The plan captures the fixed execution order of segments and provides
    metadata for the streaming controller to make scheduling decisions.

    Attributes:
        segments: Ordered list of segments matching forward execution order.
        double_block_count: Total number of DoubleStreamBlocks.
        single_block_count: Total number of SingleStreamBlocks.
    """

    segments: List[Segment] = field(default_factory=list)
    double_block_count: int = 0
    single_block_count: int = 0

    @property
    def total_bytes(self) -> int:
        """Total parameter memory across all segments."""
        return sum(seg.param_bytes for seg in self.segments)

    @property
    def total_blocks(self) -> int:
        """Total number of blocks across all segments."""
        return self.double_block_count + self.single_block_count

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)


def calculate_module_bytes(module: nn.Module) -> int:
    """Calculate total parameter memory for a module in bytes.

    This includes all parameters (trainable and non-trainable) but
    excludes buffers to match typical VRAM accounting.
    """
    total = 0
    for param in module.parameters():
        total += param.numel() * param.element_size()
    return total
