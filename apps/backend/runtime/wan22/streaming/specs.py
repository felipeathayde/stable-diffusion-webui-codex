"""Core dataclasses for WAN streaming infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


@dataclass
class WanBlockInfo:
    """Metadata for a single WAN transformer block.

    Unlike Flux streaming which tracks nn.Module references, WAN streaming
    tracks tensor keys that need to be loaded for each block.

    Attributes:
        index: Block index in the model.
        tensor_keys: Set of tensor keys in state dict for this block.
        param_bytes: Estimated parameter memory in bytes.
    """

    index: int
    tensor_keys: Set[str] = field(default_factory=set)
    param_bytes: int = 0

    def __hash__(self) -> int:
        return hash(self.index)


@dataclass
class WanExecutionPlan:
    """Execution plan for streaming WAN transformer blocks.

    Attributes:
        blocks: Ordered list of WanBlockInfo matching forward execution order.
        total_bytes: Total parameter memory across all blocks.
        block_count: Number of transformer blocks.
    """

    blocks: List[WanBlockInfo] = field(default_factory=list)
    total_bytes: int = 0
    block_count: int = 0

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)


def calculate_tensor_bytes(tensor: Any) -> int:
    """Calculate memory size of a tensor in bytes.

    Handles both regular PyTorch tensors and GGUF quantized tensors.
    """
    if tensor is None:
        return 0

    # Check for GGUF quantized tensor
    if hasattr(tensor, "gguf_cls"):
        # GGUF tensors store quantized data; estimate from data attribute
        if hasattr(tensor, "data") and hasattr(tensor.data, "numel"):
            return tensor.data.numel() * tensor.data.element_size()
        # Fallback: use shape and assume average 4 bits per element
        if hasattr(tensor, "shape"):
            import math
            return math.ceil(torch.Size(tensor.shape).numel() * 0.5)

    # Regular PyTorch tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.numel() * tensor.element_size()

    return 0


def extract_block_tensor_keys(block_index: int, state_keys: List[str]) -> Set[str]:
    """Extract all tensor keys belonging to a specific block.

    WAN blocks have keys like:
    - blocks.{i}.self_attn.q.weight
    - blocks.{i}.cross_attn.k.weight
    - blocks.{i}.ffn.0.weight
    - blocks.{i}.modulation.weight
    etc.
    """
    prefix = f"blocks.{block_index}."
    return {k for k in state_keys if k.startswith(prefix)}


def build_execution_plan(
    state: Dict[str, Any],
    n_blocks: int,
) -> WanExecutionPlan:
    """Build an execution plan from WAN model state dict.

    Args:
        state: Model state dictionary with tensor keys.
        n_blocks: Number of transformer blocks in the model.

    Returns:
        WanExecutionPlan with ordered blocks matching forward execution.
    """
    all_keys = list(state.keys())
    blocks: List[WanBlockInfo] = []
    total_bytes = 0

    for i in range(n_blocks):
        tensor_keys = extract_block_tensor_keys(i, all_keys)
        param_bytes = sum(calculate_tensor_bytes(state.get(k)) for k in tensor_keys)

        block = WanBlockInfo(
            index=i,
            tensor_keys=tensor_keys,
            param_bytes=param_bytes,
        )
        blocks.append(block)
        total_bytes += param_bytes

    return WanExecutionPlan(
        blocks=blocks,
        total_bytes=total_bytes,
        block_count=n_blocks,
    )

