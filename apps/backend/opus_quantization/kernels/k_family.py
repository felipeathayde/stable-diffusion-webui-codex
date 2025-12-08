# OpusQuantization - K-family quantization kernels (Q2_K to Q6_K)

from __future__ import annotations

from typing import Any, Tuple, TYPE_CHECKING

import numpy as np
import torch

from ..core import QuantType, register_quant

if TYPE_CHECKING:
    from ..tensor import OpusParameter

__all__ = ["register_k_family_quants"]


# Constants
QK_K = 256  # Elements per K-family super-block
K_SCALE_SIZE = 12  # Size of scale/min encoding for Q4_K/Q5_K


# =============================================================================
# Utility functions for K-family
# =============================================================================

def _get_scale_min_numpy(scales: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract scale and min values from K-family scale encoding."""
    n_blocks = scales.shape[0]
    scales = scales.view(np.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = np.split(scales, 3, axis=-2)
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return (sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8)))


def _get_scale_min_torch(scales: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract scale and min values from K-family scale encoding (torch)."""
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return (sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8)))


# =============================================================================
# Q2_K: 2-bit K-family quantization
# Block: 16 (scales) + 64 (qs) + 2 (d) + 2 (dmin) = 84 bytes
# =============================================================================

def _q2_k_bake(param: "OpusParameter") -> None:
    """Q2_K doesn't need special baking."""
    pass


def _q2_k_dequantize(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize Q2_K blocks."""
    n_blocks = blocks.shape[0]
    device = blocks.device
    
    scales = blocks[..., :QK_K // 16]
    qs = blocks[..., QK_K // 16:QK_K // 16 + QK_K // 4]
    d = blocks[..., -4:-2]
    dmin = blocks[..., -2:]
    
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    
    shift = torch.tensor([0, 2, 4, 6], device=device, dtype=torch.uint8).reshape(1, 1, 4, 1)
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16)).to(dtype)
    
    return (dl * qs - ml).reshape((n_blocks, -1))


def _q2_k_dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    """Dequantize Q2_K blocks (numpy)."""
    n_blocks = blocks.shape[0]
    scales, rest = np.hsplit(blocks, [QK_K // 16])
    qs, rest = np.hsplit(rest, [QK_K // 4])
    d, dmin = np.hsplit(rest, [2])

    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)

    dl = (d * (scales & 0xF).astype(np.float32)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4).astype(np.float32)).reshape((n_blocks, QK_K // 16, 1))

    shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & np.uint8(3)
    qs = qs.reshape((n_blocks, QK_K // 16, 16)).astype(np.float32)
    return (dl * qs - ml).reshape((n_blocks, -1))


# =============================================================================
# Q3_K: 3-bit K-family quantization
# Block: 32 (hmask) + 64 (qs) + 12 (scales) + 2 (d) = 110 bytes
# =============================================================================

def _q3_k_bake(param: "OpusParameter") -> None:
    """Q3_K doesn't need special baking."""
    pass


def _q3_k_dequantize(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize Q3_K blocks."""
    n_blocks = blocks.shape[0]
    device = blocks.device
    
    hmask = blocks[..., :QK_K // 8]
    qs = blocks[..., QK_K // 8:QK_K // 8 + QK_K // 4]
    scales = blocks[..., QK_K // 8 + QK_K // 4:-2]
    d = blocks[..., -2:]
    
    d = d.view(torch.float16).to(dtype)
    
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)
    dl = (d * scales).reshape((n_blocks, 16, 1))
    
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor(list(range(8)), device=device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))
    
    return (dl * q).reshape((n_blocks, QK_K))


def _q3_k_dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    """Dequantize Q3_K blocks (numpy)."""
    n_blocks = blocks.shape[0]

    hmask, rest = np.hsplit(blocks, [QK_K // 8])
    qs, rest = np.hsplit(rest, [QK_K // 4])
    scales, d = np.hsplit(rest, [12])

    d = d.view(np.float16).astype(np.float32)

    lscales, hscales = np.hsplit(scales, [8])
    lscales = lscales.reshape((n_blocks, 1, 8)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & np.uint8(0x0F)) | ((hscales & np.uint8(0x03)) << np.uint8(4))
    scales = (scales.astype(np.int8) - np.int8(32)).astype(np.float32)

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> np.array(list(range(8)), dtype=np.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(3)
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(1))
    qh = qh ^ np.uint8(1)
    q = (ql.astype(np.int8) - (qh << np.uint8(2)).astype(np.int8)).astype(np.float32)

    return (dl * q).reshape((n_blocks, QK_K))


# =============================================================================
# Q4_K: 4-bit K-family quantization
# Block: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = 144 bytes
# =============================================================================

def _q4_k_bake(param: "OpusParameter") -> None:
    """Prepare Q4_K tensor for dequantization.
    
    Note: We do NOT convert the scale dtype here because computation_dtype
    may not be set correctly yet. The dequantize function handles conversion.
    """
    # No conversion needed - data is already in GGUF format
    # The dequantize function reads scales as float16 and converts to target dtype
    pass


def _q4_k_dequantize(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize Q4_K blocks."""
    n_blocks = blocks.shape[0]
    device = blocks.device
    
    d = blocks[..., :2]
    dmin = blocks[..., 2:4]
    scales = blocks[..., 4:4 + K_SCALE_SIZE]
    qs = blocks[..., 4 + K_SCALE_SIZE:]
    
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, mn = _get_scale_min_torch(scales, device)
    d = (d * sc.to(dtype)).reshape((n_blocks, -1, 1))
    dm = (dmin * mn.to(dtype)).reshape((n_blocks, -1, 1))
    
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32)).to(dtype)
    
    return (d * qs - dm).reshape((n_blocks, QK_K))


def _q4_k_dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    """Dequantize Q4_K blocks (numpy)."""
    n_blocks = blocks.shape[0]
    d, rest = np.hsplit(blocks, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, qs = np.hsplit(rest, [K_SCALE_SIZE])

    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)

    sc, mn = _get_scale_min_numpy(scales)

    d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
    dm = (dmin * mn.astype(np.float32)).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 32)).astype(np.float32)

    return (d * qs - dm).reshape((n_blocks, QK_K))


# =============================================================================
# Q5_K: 5-bit K-family quantization
# Block: 2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (qs) = 176 bytes
# =============================================================================

def _q5_k_bake(param: "OpusParameter") -> None:
    """Q5_K doesn't need special baking."""
    pass


def _q5_k_dequantize(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize Q5_K blocks."""
    n_blocks = blocks.shape[0]
    device = blocks.device
    
    d = blocks[..., :2]
    dmin = blocks[..., 2:4]
    scales = blocks[..., 4:4 + K_SCALE_SIZE]
    qh = blocks[..., 4 + K_SCALE_SIZE:4 + K_SCALE_SIZE + QK_K // 8]
    qs = blocks[..., 4 + K_SCALE_SIZE + QK_K // 8:]
    
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, mn = _get_scale_min_torch(scales, device)
    d = (d * sc.to(dtype)).reshape((n_blocks, -1, 1))
    dm = (dmin * mn.to(dtype)).reshape((n_blocks, -1, 1))
    
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh_bits = qh.reshape((n_blocks, -1, 1, 32)) >> torch.arange(8, device=device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh_bits = (qh_bits & 1).reshape((n_blocks, -1, 32))
    q = (ql | (qh_bits << 4)).to(dtype)
    
    return (d * q - dm).reshape((n_blocks, QK_K))


def _q5_k_dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    """Dequantize Q5_K blocks (numpy)."""
    n_blocks = blocks.shape[0]
    d, rest = np.hsplit(blocks, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, rest = np.hsplit(rest, [K_SCALE_SIZE])
    qh, qs = np.hsplit(rest, [QK_K // 8])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)
    sc, mn = _get_scale_min_numpy(scales)
    d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
    dm = (dmin * mn.astype(np.float32)).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh_bits = qh.reshape((n_blocks, -1, 1, 32)) >> np.arange(8, dtype=np.uint8).reshape((1, 1, 8, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
    qh_bits = (qh_bits & np.uint8(1)).reshape((n_blocks, -1, 32))
    q = (ql | (qh_bits << np.uint8(4))).astype(np.float32)
    return (d * q - dm).reshape((n_blocks, QK_K))


# =============================================================================
# Q6_K: 6-bit K-family quantization
# Block: 128 (ql) + 64 (qh) + 16 (scales) + 2 (d) = 210 bytes
# =============================================================================

def _q6_k_bake(param: "OpusParameter") -> None:
    """Q6_K doesn't need special baking."""
    pass


def _q6_k_dequantize(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize Q6_K blocks."""
    n_blocks = blocks.shape[0]
    device = blocks.device
    
    ql = blocks[..., :QK_K // 2]
    qh = blocks[..., QK_K // 2:QK_K // 2 + QK_K // 4]
    scales = blocks[..., QK_K // 2 + QK_K // 4:-2]
    d = blocks[..., -2:]
    
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(dtype) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    
    return (d * q).reshape((n_blocks, QK_K))


def _q6_k_dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    """Dequantize Q6_K blocks (numpy)."""
    n_blocks = blocks.shape[0]
    ql, rest = np.hsplit(blocks, [QK_K // 2])
    qh, rest = np.hsplit(rest, [QK_K // 4])
    scales, d = np.hsplit(rest, [QK_K // 16])

    scales = scales.view(np.int8).astype(np.float32)
    d = d.view(np.float16).astype(np.float32)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = (qh & np.uint8(0x03)).reshape((n_blocks, -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
    q = q.reshape((n_blocks, QK_K // 16, -1)).astype(np.float32)

    return (d * q).reshape((n_blocks, QK_K))


# =============================================================================
# Registration
# =============================================================================

def register_k_family_quants() -> None:
    """Register all K-family quantization types."""
    
    register_quant(
        QuantType.Q2_K,
        dequantize=_q2_k_dequantize,
        bake=_q2_k_bake,
        dequantize_numpy=_q2_k_dequantize_numpy,
        description="2-bit K-family quantization",
        requires_bake=False,
    )
    
    register_quant(
        QuantType.Q3_K,
        dequantize=_q3_k_dequantize,
        bake=_q3_k_bake,
        dequantize_numpy=_q3_k_dequantize_numpy,
        description="3-bit K-family quantization",
        requires_bake=False,
    )
    
    register_quant(
        QuantType.Q4_K,
        dequantize=_q4_k_dequantize,
        bake=_q4_k_bake,
        dequantize_numpy=_q4_k_dequantize_numpy,
        description="4-bit K-family quantization",
    )
    
    register_quant(
        QuantType.Q5_K,
        dequantize=_q5_k_dequantize,
        bake=_q5_k_bake,
        dequantize_numpy=_q5_k_dequantize_numpy,
        description="5-bit K-family quantization",
        requires_bake=False,
    )
    
    register_quant(
        QuantType.Q6_K,
        dequantize=_q6_k_dequantize,
        bake=_q6_k_bake,
        dequantize_numpy=_q6_k_dequantize_numpy,
        description="6-bit K-family quantization",
        requires_bake=False,
    )


# Auto-register on import
register_k_family_quants()
