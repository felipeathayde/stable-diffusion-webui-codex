# OpusQuantization - Core types, registry, and base classes

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, Optional, Tuple, Any

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "QuantType",
    "BLOCK_SIZES",
    "QuantSpec",
    "register_quant",
    "get_quant_spec",
    "QUANT_REGISTRY",
]


class QuantType(IntEnum):
    """
    All supported GGML quantization types.
    Values match GGML's GGMLQuantizationType enum.
    """
    # Basic family (block_size=32)
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    
    # K-family (block_size=256, super-blocks)
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    
    # Special
    F16 = 1
    F32 = 0
    BF16 = 30
    
    # IQ-family (future)
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ2_S = 18
    IQ3_XXS = 19
    IQ3_S = 20
    IQ1_S = 24
    IQ1_M = 29
    IQ4_NL = 26
    IQ4_XS = 27


# Block sizes: (elements_per_block, bytes_per_block)
BLOCK_SIZES: Dict[QuantType, Tuple[int, int]] = {
    # Basic family: 32 elements per block
    QuantType.Q4_0: (32, 18),   # 2 (scale) + 16 (4-bit packed)
    QuantType.Q4_1: (32, 20),   # 2 (scale) + 2 (min) + 16 (4-bit packed)
    QuantType.Q5_0: (32, 22),   # 2 (scale) + 4 (high bits) + 16 (low 4-bit)
    QuantType.Q5_1: (32, 24),   # 2 (scale) + 2 (min) + 4 (high bits) + 16 (low 4-bit)
    QuantType.Q8_0: (32, 34),   # 2 (scale) + 32 (int8)
    
    # K-family: 256 elements per block (super-blocks)
    QuantType.Q2_K: (256, 84),
    QuantType.Q3_K: (256, 110),
    QuantType.Q4_K: (256, 144),
    QuantType.Q5_K: (256, 176),
    QuantType.Q6_K: (256, 210),
    
    # Special
    QuantType.F16: (1, 2),
    QuantType.F32: (1, 4),
    QuantType.BF16: (1, 2),
}


# Type aliases for kernel functions
DequantizeFn = Callable[[torch.Tensor, torch.dtype], torch.Tensor]
QuantizeFn = Callable[[torch.Tensor, Any], torch.Tensor]
BakeFn = Callable[[Any], None]
DequantizeNumpyFn = Callable  # For numpy compatibility


@dataclass
class QuantSpec:
    """Specification for a quantization type."""
    qtype: QuantType
    block_size: int
    type_size: int
    
    # PyTorch kernels
    dequantize: Optional[DequantizeFn] = None
    quantize: Optional[QuantizeFn] = None
    bake: Optional[BakeFn] = None
    
    # NumPy kernels (for compatibility)
    dequantize_numpy: Optional[DequantizeNumpyFn] = None
    quantize_numpy: Optional[DequantizeNumpyFn] = None
    
    # Metadata
    description: str = ""
    requires_bake: bool = True


# Global registry
QUANT_REGISTRY: Dict[QuantType, QuantSpec] = {}


def register_quant(
    qtype: QuantType,
    *,
    dequantize: Optional[DequantizeFn] = None,
    quantize: Optional[QuantizeFn] = None,
    bake: Optional[BakeFn] = None,
    dequantize_numpy: Optional[DequantizeNumpyFn] = None,
    quantize_numpy: Optional[DequantizeNumpyFn] = None,
    description: str = "",
    requires_bake: bool = True,
) -> None:
    """Register a quantization type with its kernels."""
    if qtype not in BLOCK_SIZES:
        raise ValueError(f"Unknown quant type: {qtype}")
    
    block_size, type_size = BLOCK_SIZES[qtype]
    
    spec = QuantSpec(
        qtype=qtype,
        block_size=block_size,
        type_size=type_size,
        dequantize=dequantize,
        quantize=quantize,
        bake=bake,
        dequantize_numpy=dequantize_numpy,
        quantize_numpy=quantize_numpy,
        description=description or qtype.name,
        requires_bake=requires_bake,
    )
    
    QUANT_REGISTRY[qtype] = spec
    logger.debug("Registered quant type: %s (block=%d, bytes=%d)", qtype.name, block_size, type_size)


def get_quant_spec(qtype: QuantType) -> Optional[QuantSpec]:
    """Get the specification for a quantization type."""
    return QUANT_REGISTRY.get(qtype)
