from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ....quick_4bits_ops import quick_unpack_4bits_u

__all__ = [
    "quantize_numpy",
    "dequantize_numpy",
    "quantize_torch",
    "dequantize_torch",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — Q4_1
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


def quantize_numpy(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    block_max = blocks.max(axis=-1, keepdims=True)
    block_min = blocks.min(axis=-1, keepdims=True)
    d = (block_max - block_min) / 15.0
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, 0, 1.0 / d)
    qs = np.trunc((blocks - block_min) * inv + np.float32(0.5)).astype(np.uint8)
    qs = np.clip(qs, 0, 15)
    qs = qs.reshape((n_blocks, 2, blocks.shape[-1] // 2))
    packed = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))
    header_d = d.astype(np.float16).view(np.uint8)
    header_m = block_min.astype(np.float16).view(np.uint8)
    return np.concatenate([header_d, header_m, packed], axis=-1)


def dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    header_d, remainder = np.hsplit(blocks, [2])
    header_m, payload = np.hsplit(remainder, [2])
    d = header_d.view(np.float16).astype(np.float32)
    m = header_m.view(np.float16).astype(np.float32)
    reshaped = payload.reshape((blocks.shape[0], -1, 1, payload.shape[-1]))
    qs = reshaped >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((blocks.shape[0], -1)).astype(np.float32)
    return (d * qs) + m


def quantize_torch(blocks: torch.Tensor, parent: Any) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    block_max = blocks.max(dim=-1, keepdim=True).values
    block_min = blocks.min(dim=-1, keepdim=True).values
    d = (block_max - block_min) / 15.0
    inv = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    q = torch.trunc((blocks - block_min) * inv + 0.5).clamp(0, 15).to(torch.uint8)
    q = q.reshape((n_blocks, 2, blocks.shape[-1] // 2))
    packed = (q[..., 0, :] & 0x0F) | (q[..., 1, :] << 4)
    header_d = d.to(parent.computation_dtype).view(torch.uint8)
    header_m = block_min.to(parent.computation_dtype).view(torch.uint8)
    return torch.cat([header_d, header_m, packed], dim=-1)


def dequantize_torch(blocks: torch.Tensor, parent: Any) -> torch.Tensor:
    header_d = blocks[..., :2]
    header_m = blocks[..., 2:4]
    payload = blocks[..., 4:]
    d = header_d.view(parent.computation_dtype)
    m = header_m.view(parent.computation_dtype)
    qs = quick_unpack_4bits_u(payload)
    return (d * qs) + m
