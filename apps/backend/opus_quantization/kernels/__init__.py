# OpusQuantization - Kernels package
# Registers all quantization types using dequant.py (ported from ComfyUI-GGUF)

from __future__ import annotations

import functools
import torch
from typing import Optional

from ..core import QuantType, register_quant, BLOCK_SIZES
from ..dequant import (
    # Legacy quants
    dequantize_blocks_BF16,
    dequantize_blocks_Q8_0,
    dequantize_blocks_Q5_1,
    dequantize_blocks_Q5_0,
    dequantize_blocks_Q4_1,
    dequantize_blocks_Q4_0,
    # K-quants
    dequantize_blocks_Q6_K,
    dequantize_blocks_Q5_K,
    dequantize_blocks_Q4_K,
    dequantize_blocks_Q3_K,
    dequantize_blocks_Q2_K,
    # IQ quants
    dequantize_blocks_IQ4_NL,
    dequantize_blocks_IQ4_XS,
)


def _make_dequantize_wrapper(dequant_fn, block_size: int, type_size: int):
    """Create a wrapper that matches the expected DequantizeFn signature.
    
    The dequant.py functions have signature:
        dequant_fn(blocks, block_size, type_size, dtype) -> Tensor
    
    The registry expects:
        dequantize(blocks, dtype) -> Tensor
    
    This wrapper binds block_size and type_size, and reshapes the tensor
    to (n_blocks, type_size) as expected by dequant functions.
    
    Reference: city96/ComfyUI-GGUF dequant.py::dequantize()
    """
    @functools.wraps(dequant_fn)
    def wrapper(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # Reshape to (n_blocks, type_size) as expected by dequant functions
        # Reference: ComfyUI-GGUF dequant.py lines 28-32
        rows = blocks.reshape((-1, blocks.shape[-1])).view(torch.uint8)
        n_blocks = rows.numel() // type_size
        reshaped = rows.reshape((n_blocks, type_size))
        return dequant_fn(reshaped, block_size, type_size, dtype)
    return wrapper


def _noop_bake(param) -> None:
    """No-op bake function for new dequant.py functions.
    
    The new dequantization functions don't require pre-processing (bake).
    They read scale/min values directly as float16 and convert during dequant.
    """
    pass


def _register_all() -> None:
    """Register all quantization types with their kernels."""
    
    # Define all quant types and their dequant functions
    QUANT_DEFS = [
        # Legacy quants
        (QuantType.Q4_0, dequantize_blocks_Q4_0, "4-bit zero-centered"),
        (QuantType.Q4_1, dequantize_blocks_Q4_1, "4-bit with min offset"),
        (QuantType.Q5_0, dequantize_blocks_Q5_0, "5-bit zero-centered"),
        (QuantType.Q5_1, dequantize_blocks_Q5_1, "5-bit with min offset"),
        (QuantType.Q8_0, dequantize_blocks_Q8_0, "8-bit"),
        (QuantType.BF16, dequantize_blocks_BF16, "BFloat16"),
        
        # K-quants
        (QuantType.Q2_K, dequantize_blocks_Q2_K, "2-bit K-quant"),
        (QuantType.Q3_K, dequantize_blocks_Q3_K, "3-bit K-quant"),
        (QuantType.Q4_K, dequantize_blocks_Q4_K, "4-bit K-quant"),
        (QuantType.Q5_K, dequantize_blocks_Q5_K, "5-bit K-quant"),
        (QuantType.Q6_K, dequantize_blocks_Q6_K, "6-bit K-quant"),
        
        # IQ quants
        (QuantType.IQ4_NL, dequantize_blocks_IQ4_NL, "IQ4 non-linear"),
        (QuantType.IQ4_XS, dequantize_blocks_IQ4_XS, "IQ4 extra-small"),
    ]
    
    for qtype, dequant_fn, desc in QUANT_DEFS:
        if qtype not in BLOCK_SIZES:
            continue  # Skip if not defined in BLOCK_SIZES
            
        block_size, type_size = BLOCK_SIZES[qtype]
        
        register_quant(
            qtype,
            dequantize=_make_dequantize_wrapper(dequant_fn, block_size, type_size),
            bake=_noop_bake,
            description=desc,
            requires_bake=False,  # New functions don't need bake
        )


# Register on import
_register_all()

__all__ = []
