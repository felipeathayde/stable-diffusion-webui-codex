# OpusQuantization - Basic quantization kernels (Q4, Q5, Q8, BF16)

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from ..core import QuantType, register_quant
from ..ops import unpack_4bits_signed, unpack_4bits_unsigned, split_blocks, reorder_4bits

if TYPE_CHECKING:
    from ..tensor import OpusParameter

__all__ = ["register_basic_quants"]


# =============================================================================
# Q4_0: 4-bit quantization, zero-centered
# Block: 2 bytes (scale) + 16 bytes (32 x 4-bit packed) = 18 bytes
# =============================================================================

def _q4_0_bake(param: "OpusParameter") -> None:
    """Prepare Q4_0 tensor for dequantization.
    
    Note: We do NOT convert the scale dtype here because computation_dtype
    may not be set correctly yet (it's updated during forward pass).
    The scale remains as float16 (original GGUF format) and is converted
    during dequantize.
    """
    # No conversion needed - data is already in GGUF format
    # Just mark as baked
    pass


def _q4_0_dequantize(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize Q4_0 blocks.
    
    Q4_0 format: 2 bytes (scale as dtype) + 16 bytes (32 x 4-bit packed) = 18 bytes per block
    After bake, scale is already in computation_dtype.
    
    The input tensor can be:
    - 1D: (total_bytes,) - flat array
    - 2D: (rows, bytes_per_row) - matrix where each row has multiple blocks
    
    Uses optimized lookup table for fast 4-bit unpacking.
    """
    BLOCK_SIZE = 18  # bytes per Q4_0 block
    ELEMENTS_PER_BLOCK = 32
    
    original_shape = blocks.shape
    
    # Flatten to 1D for uniform processing
    if blocks.dim() == 1:
        flat = blocks
        output_shape = None
    else:
        # For 2D: (rows, bytes_per_row) -> need to output (rows, elements_per_row)
        rows = blocks.shape[0]
        bytes_per_row = blocks.shape[1]
        blocks_per_row = bytes_per_row // BLOCK_SIZE
        elements_per_row = blocks_per_row * ELEMENTS_PER_BLOCK
        output_shape = (rows, elements_per_row)
        flat = blocks.reshape(-1)
    
    # Calculate number of blocks
    total_bytes = flat.numel()
    n_blocks = total_bytes // BLOCK_SIZE
    
    # Reshape to (n_blocks, 18)
    blocked = flat[:n_blocks * BLOCK_SIZE].view(n_blocks, BLOCK_SIZE)
    
    # Extract scale (2 bytes as float16) and quantized values (16 bytes)
    # Scale is stored as float16 in original GGUF format - convert to target dtype
    d = blocked[:, :2].contiguous().view(torch.float16).to(dtype)  # (n_blocks, 1)
    qs = blocked[:, 2:]  # (n_blocks, 16)
    
    # Unpack 4-bit values: 16 bytes -> 32 int8 values per block
    unpacked = unpack_4bits_signed(qs)  # (n_blocks, 32)
    
    # Dequantize
    result = d * unpacked.to(dtype)  # (n_blocks, 32)
    
    # Reshape to output shape
    if output_shape is not None:
        result = result.view(output_shape)
    else:
        result = result.flatten()
    
    return result


def _q4_0_dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    """Dequantize Q4_0 blocks (numpy)."""
    header, payload = np.hsplit(blocks, [2])
    d = header.view(np.float16).astype(np.float32)
    reshaped = payload.reshape((blocks.shape[0], -1, 1, payload.shape[-1]))
    qs = reshaped >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((blocks.shape[0], -1)).astype(np.int8) - np.int8(8)
    return d * qs.astype(np.float32)


# =============================================================================
# Q4_1: 4-bit quantization with min value
# Block: 2 (scale) + 2 (min) + 16 (packed) = 20 bytes
# =============================================================================

def _q4_1_bake(param: "OpusParameter") -> None:
    """Prepare Q4_1 tensor for dequantization.
    
    Note: We do NOT convert the scale/min dtype here because computation_dtype
    may not be set correctly yet. Values remain as float16 (original GGUF format).
    """
    # No conversion needed - data is already in GGUF format
    pass


def _q4_1_dequantize(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize Q4_1 blocks.
    
    Q4_1 format: 2 (scale) + 2 (min) + 16 (packed) = 20 bytes per block
    
    The input tensor can be 1D or 2D.
    Uses optimized lookup table for fast 4-bit unpacking.
    """
    BLOCK_SIZE = 20  # bytes per Q4_1 block
    ELEMENTS_PER_BLOCK = 32
    
    # Flatten to 1D for uniform processing
    if blocks.dim() == 1:
        flat = blocks
        output_shape = None
    else:
        rows = blocks.shape[0]
        bytes_per_row = blocks.shape[1]
        blocks_per_row = bytes_per_row // BLOCK_SIZE
        elements_per_row = blocks_per_row * ELEMENTS_PER_BLOCK
        output_shape = (rows, elements_per_row)
        flat = blocks.reshape(-1)
    
    total_bytes = flat.numel()
    n_blocks = total_bytes // BLOCK_SIZE
    blocked = flat[:n_blocks * BLOCK_SIZE].view(n_blocks, BLOCK_SIZE)
    
    # Scale and min are stored as float16 in original GGUF format - convert to target dtype
    d = blocked[:, :2].contiguous().view(torch.float16).to(dtype)
    m = blocked[:, 2:4].contiguous().view(torch.float16).to(dtype)
    qs = blocked[:, 4:]  # 16 bytes
    
    unpacked = unpack_4bits_unsigned(qs)
    result = (d * unpacked.to(dtype)) + m
    
    if output_shape is not None:
        result = result.view(output_shape)
    else:
        result = result.flatten()
    
    return result


def _q4_1_dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    """Dequantize Q4_1 blocks (numpy)."""
    header_d, remainder = np.hsplit(blocks, [2])
    header_m, payload = np.hsplit(remainder, [2])
    d = header_d.view(np.float16).astype(np.float32)
    m = header_m.view(np.float16).astype(np.float32)
    reshaped = payload.reshape((blocks.shape[0], -1, 1, payload.shape[-1]))
    qs = reshaped >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((blocks.shape[0], -1)).astype(np.float32)
    return (d * qs) + m


# =============================================================================
# Q5_0: 5-bit quantization, zero-centered
# Block: 2 (scale) + 4 (high bits) + 16 (low 4-bit) = 22 bytes
# =============================================================================

def _q5_0_bake(param: "OpusParameter") -> None:
    """Prepare Q5_0 tensor for dequantization.
    
    Note: We do NOT convert the scale dtype here because computation_dtype
    may not be set correctly yet. Scale remains as float16 (original GGUF format).
    """
    # No conversion needed - data is already in GGUF format
    pass


def _q5_0_dequantize(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize Q5_0 blocks.
    
    Q5_0 format: 2 (scale) + 4 (high bits) + 16 (low 4-bit) = 22 bytes per block
    
    The input tensor can be 1D or 2D.
    """
    BLOCK_SIZE = 22  # bytes per Q5_0 block
    ELEMENTS_PER_BLOCK = 32
    
    # Flatten to 1D for uniform processing
    if blocks.dim() == 1:
        flat = blocks
        output_shape = None
    else:
        rows = blocks.shape[0]
        bytes_per_row = blocks.shape[1]
        blocks_per_row = bytes_per_row // BLOCK_SIZE
        elements_per_row = blocks_per_row * ELEMENTS_PER_BLOCK
        output_shape = (rows, elements_per_row)
        flat = blocks.reshape(-1)
    
    total_bytes = flat.numel()
    n_blocks = total_bytes // BLOCK_SIZE
    blocked = flat[:n_blocks * BLOCK_SIZE].view(n_blocks, BLOCK_SIZE)
    
    # Split: scale(2) + high_bits(4) + low_bits(16)
    # Scale is stored as float16 in original GGUF format - convert to target dtype
    d = blocked[:, :2].contiguous().view(torch.float16).to(dtype)
    qh = blocked[:, 2:6]
    qs = blocked[:, 6:]
    
    # Extract high bits (32 bits total, one per element)
    qh_uint32 = qh.contiguous().view(torch.uint8).view(torch.uint32)
    # Use int64 for bit positions to avoid CPU uint32 limitation
    bit_positions = torch.arange(32, device=blocks.device, dtype=torch.int64).reshape(1, 32)
    # Cast qh to int64 for shift operation
    qh_int64 = qh_uint32.reshape(n_blocks, 1).to(torch.int64)
    high = ((qh_int64 >> bit_positions) & 1).to(torch.uint8)
    
    # Unpack low 4 bits using lookup table
    low = unpack_4bits_unsigned(qs)
    
    # Combine: value = (low | (high << 4)) - 16
    values = (low | (high << 4)).to(torch.int8) - 16
    result = d * values.to(dtype)
    
    if output_shape is not None:
        result = result.view(output_shape)
    else:
        result = result.flatten()
    
    return result


def _q5_0_dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    """Dequantize Q5_0 blocks (numpy)."""
    d, rest = np.hsplit(blocks, [2])
    qh, qs = np.hsplit(rest, [4])
    d = d.view(np.float16).astype(np.float32)
    qh = qh.view(np.uint32)
    qh = qh.reshape((blocks.shape[0], 1)) >> np.arange(32, dtype=np.uint32).reshape((1, 32))
    qh = (qh & np.uint32(0x01)).astype(np.uint8)
    ql = qs.reshape((blocks.shape[0], -1, 1, qs.shape[-1])) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((blocks.shape[0], -1))
    qs_vals = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(16)
    return d * qs_vals.astype(np.float32)


# =============================================================================
# Q5_1: 5-bit quantization with min value
# Block: 2 (scale) + 2 (min) + 4 (high bits) + 16 (low 4-bit) = 24 bytes
# =============================================================================

def _q5_1_bake(param: "OpusParameter") -> None:
    """Prepare Q5_1 tensor for dequantization.
    
    Note: We do NOT convert the scale/min dtype here because computation_dtype
    may not be set correctly yet. Values remain as float16 (original GGUF format).
    """
    # No conversion needed - data is already in GGUF format
    pass


def _q5_1_dequantize(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize Q5_1 blocks.
    
    Q5_1 format: 2 (scale) + 2 (min) + 4 (high bits) + 16 (low 4-bit) = 24 bytes per block
    
    The input tensor can be 1D or 2D.
    """
    BLOCK_SIZE = 24  # bytes per Q5_1 block
    ELEMENTS_PER_BLOCK = 32
    
    # Flatten to 1D for uniform processing
    if blocks.dim() == 1:
        flat = blocks
        output_shape = None
    else:
        rows = blocks.shape[0]
        bytes_per_row = blocks.shape[1]
        blocks_per_row = bytes_per_row // BLOCK_SIZE
        elements_per_row = blocks_per_row * ELEMENTS_PER_BLOCK
        output_shape = (rows, elements_per_row)
        flat = blocks.reshape(-1)
    
    total_bytes = flat.numel()
    n_blocks = total_bytes // BLOCK_SIZE
    blocked = flat[:n_blocks * BLOCK_SIZE].view(n_blocks, BLOCK_SIZE)
    
    # Scale and min are stored as float16 in original GGUF format - convert to target dtype
    d = blocked[:, :2].contiguous().view(torch.float16).to(dtype)
    m = blocked[:, 2:4].contiguous().view(torch.float16).to(dtype)
    qh = blocked[:, 4:8]
    qs = blocked[:, 8:]
    
    # Extract high bits
    qh_uint32 = qh.contiguous().view(torch.uint8).view(torch.uint32)
    bit_positions = torch.arange(32, device=blocks.device, dtype=torch.int64).reshape(1, 32)
    qh_int64 = qh_uint32.reshape(n_blocks, 1).to(torch.int64)
    high = ((qh_int64 >> bit_positions) & 1).to(torch.uint8)
    
    # Unpack low 4 bits
    low = unpack_4bits_unsigned(qs)
    
    # Combine: value = (low | (high << 4))
    values = (low | (high << 4)).to(dtype)
    result = (d * values) + m
    
    if output_shape is not None:
        result = result.view(output_shape)
    else:
        result = result.flatten()
    
    return result


def _q5_1_dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    """Dequantize Q5_1 blocks (numpy)."""
    d, rest = np.hsplit(blocks, [2])
    m, rest = np.hsplit(rest, [2])
    qh, qs = np.hsplit(rest, [4])
    d = d.view(np.float16).astype(np.float32)
    m = m.view(np.float16).astype(np.float32)
    qh = qh.view(np.uint32)
    qh = (qh.reshape((blocks.shape[0], 1)) >> np.arange(32, dtype=np.uint32).reshape((1, 32))) & np.uint32(1)
    ql = qs.reshape((blocks.shape[0], -1, 1, qs.shape[-1])) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((blocks.shape[0], -1))
    values = (ql | (qh.astype(np.uint8) << np.uint8(4))).astype(np.float32)
    return (d * values) + m


# =============================================================================
# Q8_0: 8-bit quantization
# Block: 2 (scale) + 32 (int8 values) = 34 bytes
# =============================================================================

def _q8_0_bake(param: "OpusParameter") -> None:
    """Prepare Q8_0 tensor for dequantization.
    
    Note: We do NOT convert the scale dtype here because computation_dtype
    may not be set correctly yet. Scale remains as float16 (original GGUF format).
    """
    # No conversion needed - data is already in GGUF format
    pass


def _q8_0_dequantize(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize Q8_0 blocks.
    
    Q8_0 format: 2 (scale) + 32 (int8 values) = 34 bytes per block
    
    The input tensor can be 1D or 2D.
    """
    BLOCK_SIZE = 34  # bytes per Q8_0 block
    ELEMENTS_PER_BLOCK = 32
    
    # Flatten to 1D for uniform processing
    if blocks.dim() == 1:
        flat = blocks
        output_shape = None
    else:
        rows = blocks.shape[0]
        bytes_per_row = blocks.shape[1]
        blocks_per_row = bytes_per_row // BLOCK_SIZE
        elements_per_row = blocks_per_row * ELEMENTS_PER_BLOCK
        output_shape = (rows, elements_per_row)
        flat = blocks.reshape(-1)
    
    total_bytes = flat.numel()
    n_blocks = total_bytes // BLOCK_SIZE
    blocked = flat[:n_blocks * BLOCK_SIZE].view(n_blocks, BLOCK_SIZE)
    
    # Scale is stored as float16 in original GGUF format - convert to target dtype
    d = blocked[:, :2].contiguous().view(torch.float16).to(dtype)
    qs = blocked[:, 2:].contiguous().view(torch.int8).to(dtype)
    
    result = d * qs
    
    if output_shape is not None:
        result = result.view(output_shape)
    else:
        result = result.flatten()
    
    return result


def _q8_0_dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    """Dequantize Q8_0 blocks (numpy)."""
    header, payload = np.split(blocks, [2], axis=1)
    d = header.view(np.float16).astype(np.float32)
    x = payload.view(np.int8).astype(np.float32)
    return x * d


# =============================================================================
# BF16: BFloat16 (not really quantized, but handled similarly)
# =============================================================================

def _bf16_bake(param: "OpusParameter") -> None:
    """BF16 doesn't need baking."""
    pass


def _bf16_dequantize(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize BF16 blocks.
    
    BF16 format: 2 bytes per element (no blocking, just raw bfloat16)
    
    The shape is preserved, just dtype is converted.
    """
    # For BF16, the output shape is half the input bytes (2 bytes -> 1 element)
    original_shape = blocks.shape
    
    # View as bfloat16 and convert to target dtype
    # The shape automatically adjusts: (rows, bytes) -> (rows, bytes/2)
    result = blocks.view(torch.bfloat16).to(dtype)
    
    return result


def _bf16_dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    """Dequantize BF16 blocks (numpy)."""
    # NumPy doesn't support bfloat16 natively, convert via uint16
    as_uint16 = blocks.view(np.uint16)
    # Shift to float32 (bfloat16 is upper 16 bits of float32)
    as_uint32 = as_uint16.astype(np.uint32) << 16
    return as_uint32.view(np.float32)


# =============================================================================
# Registration
# =============================================================================

def register_basic_quants() -> None:
    """Register all basic quantization types."""
    
    register_quant(
        QuantType.Q4_0,
        dequantize=_q4_0_dequantize,
        bake=_q4_0_bake,
        dequantize_numpy=_q4_0_dequantize_numpy,
        description="4-bit zero-centered quantization",
    )
    
    register_quant(
        QuantType.Q4_1,
        dequantize=_q4_1_dequantize,
        bake=_q4_1_bake,
        dequantize_numpy=_q4_1_dequantize_numpy,
        description="4-bit quantization with min value",
    )
    
    register_quant(
        QuantType.Q5_0,
        dequantize=_q5_0_dequantize,
        bake=_q5_0_bake,
        dequantize_numpy=_q5_0_dequantize_numpy,
        description="5-bit zero-centered quantization",
    )
    
    register_quant(
        QuantType.Q5_1,
        dequantize=_q5_1_dequantize,
        bake=_q5_1_bake,
        dequantize_numpy=_q5_1_dequantize_numpy,
        description="5-bit quantization with min value",
    )
    
    register_quant(
        QuantType.Q8_0,
        dequantize=_q8_0_dequantize,
        bake=_q8_0_bake,
        dequantize_numpy=_q8_0_dequantize_numpy,
        description="8-bit quantization",
    )
    
    register_quant(
        QuantType.BF16,
        dequantize=_bf16_dequantize,
        bake=_bf16_bake,
        dequantize_numpy=_bf16_dequantize_numpy,
        description="BFloat16 (16-bit brain float)",
        requires_bake=False,
    )


# Auto-register on import
register_basic_quants()
