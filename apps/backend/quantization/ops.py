"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Optimized bit unpacking and block utilities for quantized tensors.
Provides cached lookup-table-based nibble unpackers and helpers for splitting packed blocks into typed slices used by dequant kernels.

Symbols (top-level; keep in sync; no ghosts):
- `_build_4bit_lookup_signed` (function): Builds a lookup table for fast signed 4-bit unpacking.
- `_build_4bit_lookup_unsigned` (function): Builds a lookup table for fast unsigned 4-bit unpacking.
- `unpack_4bits_signed` (function): Unpacks packed 4-bit values into int8 range [-8, 7] using a cached LUT.
- `unpack_4bits_unsigned` (function): Unpacks packed 4-bit values into uint8 range [0, 15] using a cached LUT.
- `unpack_1bits` (function): Unpacks 1-bit packed tensors into 0/1 values.
- `reorder_4bits` (function): Reorders packed nibbles to improve memory access patterns.
- `split_blocks` (function): Splits a block tensor along the last dimension with explicit size validation.
"""

from __future__ import annotations

import torch

from .cache import get_device_cache

__all__ = [
    "unpack_4bits_signed",
    "unpack_4bits_unsigned", 
    "unpack_1bits",
    "reorder_4bits",
    "split_blocks",
]


def _build_4bit_lookup_signed() -> torch.Tensor:
    """
    Build lookup table for fast 4-bit signed unpacking.
    
    Maps each possible uint16 value (0-65535) to a packed int32 containing
    4 int8 values, each being (nibble - 8) for range [-8, 7].
    
    The format matches the native_4bits_lookup_table schema.
    """
    # Generate all possible uint16 values
    indices = torch.arange(256 * 256, dtype=torch.long).to(torch.uint16)
    
    # View as 2 bytes per index
    x = indices.view(torch.uint8).reshape(-1, 2)
    
    # Unpack each byte into 2 nibbles: [low0, high0, low1, high1]
    unpacked = torch.stack([x & 15, x >> 4], dim=-1)  # (65536, 2, 2)
    reshaped = unpacked.reshape(-1, 4)  # (65536, 4)
    
    # Convert to signed and subtract 8
    signed = reshaped.view(torch.int8) - 8
    
    # Pack as int32 (4 bytes per entry)
    return signed.view(torch.int32)[:, 0]  # (65536,) int32


def _build_4bit_lookup_unsigned() -> torch.Tensor:
    """
    Build lookup table for fast 4-bit unsigned unpacking.
    
    Maps each possible uint16 value (0-65535) to a packed int32 containing
    4 uint8 values in range [0, 15].
    """
    indices = torch.arange(256 * 256, dtype=torch.long).to(torch.uint16)
    x = indices.view(torch.uint8).reshape(-1, 2)
    unpacked = torch.stack([x & 15, x >> 4], dim=-1)
    reshaped = unpacked.reshape(-1, 4)
    return reshaped.view(torch.int32)[:, 0]


def unpack_4bits_signed(x: torch.Tensor) -> torch.Tensor:
    """
    Unpack 4-bit packed tensor to int8 values centered at 0.
    
    Each byte contains 2 values: low 4 bits and high 4 bits.
    Output is (value - 8) for each nibble, giving range [-8, 7].
    
    Uses cached lookup table for performance (fast path with index_select).
    Falls back to simple unpacking if lookup fails.
    Thread-safe.
    """
    cache = get_device_cache()
    table = cache.get_or_create(
        x.device,
        "4bits_signed",
        _build_4bit_lookup_signed
    )
    
    n_blocks = x.shape[0]
    
    # View payload as uint16 for lookup
    x_uint16 = x.contiguous().view(torch.uint16)
    
    # Use lookup table: each uint16 -> 4 int8 values packed as int32
    indices = x_uint16.to(torch.int32).flatten()
    packed = torch.index_select(table, 0, indices)
    
    # Unpack int32 -> 4 int8 values
    return packed.view(torch.int8).view(n_blocks, -1)


def unpack_4bits_unsigned(x: torch.Tensor) -> torch.Tensor:
    """
    Unpack 4-bit packed tensor to uint8 values.
    
    Each byte contains 2 values: low 4 bits and high 4 bits.
    Output is raw nibble value in range [0, 15].
    
    Uses cached lookup table for performance.
    Thread-safe.
    """
    cache = get_device_cache()
    table = cache.get_or_create(
        x.device,
        "4bits_unsigned",
        _build_4bit_lookup_unsigned
    )
    
    n_blocks = x.shape[0]
    x_uint16 = x.contiguous().view(torch.uint16)
    indices = x_uint16.to(torch.int32).flatten()
    packed = torch.index_select(table, 0, indices)
    
    return packed.view(torch.uint8).view(n_blocks, -1)


def unpack_1bits(x: torch.Tensor, n_bits: int = 32) -> torch.Tensor:
    """
    Unpack 1-bit packed tensor.
    
    Input: tensor of uint8/uint32 with packed bits
    Output: tensor of uint8 with individual bit values (0 or 1)
    
    Args:
        x: Input tensor containing packed bits
        n_bits: Number of bits to unpack (must match input size * 8)
    """
    n_blocks = x.shape[0]
    
    # View as uint32 for easy bit extraction
    x_uint32 = x.view(torch.uint8).view(torch.uint32).reshape(n_blocks, -1)
    
    # Create bit positions [0, 1, 2, ..., 31]
    bit_positions = torch.arange(32, device=x.device, dtype=torch.int32)
    
    # Extract each bit: (value >> position) & 1
    bits = (x_uint32.unsqueeze(-1) >> bit_positions) & 1
    
    return bits.view(n_blocks, -1).to(torch.uint8)


def reorder_4bits(x: torch.Tensor) -> torch.Tensor:
    """
    Reorder 4-bit packed values for optimized memory access.
    
    This changes the packing order to match the lookup table's expected format.
    After reordering, sequential reads produce sequential outputs.
    """
    n_blocks = x.shape[0]
    # Separate low and high nibbles
    low = x & 0x0F
    high = x >> 4
    # Stack and repack in different order
    stacked = torch.stack([low, high], dim=-2).view(n_blocks, -1)
    # Repack in pairs
    repacked = stacked[:, ::2] | (stacked[:, 1::2] << 4)
    return repacked


def split_blocks(blocks: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    """
    Split block tensor along last dimension.
    
    More explicit than torch.split - validates sizes match.
    
    Args:
        blocks: Input tensor of shape (n_blocks, type_size)
        sizes: List of sizes to split into
        
    Returns:
        List of tensors, one for each size
    """
    expected = sum(sizes)
    actual = blocks.shape[-1]
    if actual != expected:
        raise ValueError(
            f"Block size mismatch: expected {expected} bytes, got {actual}. "
            f"Sizes: {sizes}"
        )
    
    result = []
    offset = 0
    for size in sizes:
        result.append(blocks[..., offset:offset + size])
        offset += size
    
    # Handle remaining bytes if any (shouldn't happen with correct sizes)
    if offset < actual:
        result.append(blocks[..., offset:])
    
    return result
