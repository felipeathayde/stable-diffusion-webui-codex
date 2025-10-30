from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ....quick_4bits_ops import quick_unpack_4bits

__all__ = [
    "quantize_numpy",
    "dequantize_numpy",
    "quantize_torch",
    "dequantize_torch",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — Q4_0
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


def quantize_numpy(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    imax = np.abs(blocks).argmax(axis=-1, keepdims=True)
    max_vals = np.take_along_axis(blocks, imax, axis=-1)
    d = max_vals / -8.0
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, 0, 1.0 / d)
    qs = np.trunc((np.float64(blocks) * np.float64(inv)) + np.float64(8.5)).astype(np.uint8)
    qs = np.clip(qs, 0, 15)
    qs = qs.reshape((n_blocks, 2, blocks.shape[-1] // 2))
    packed = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))
    header = d.astype(np.float16).view(np.uint8)
    return np.concatenate([header, packed], axis=-1)


def dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    header, payload = np.hsplit(blocks, [2])
    d = header.view(np.float16).astype(np.float32)
    reshaped = payload.reshape((blocks.shape[0], -1, 1, payload.shape[-1]))
    qs = reshaped >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((blocks.shape[0], -1)).astype(np.int8) - np.int8(8)
    return d * qs.astype(np.float32)


def quantize_torch(blocks: torch.Tensor, parent: Any) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    abs_blocks = torch.abs(blocks)
    idx = abs_blocks.argmax(dim=-1, keepdim=True)
    peak = torch.gather(blocks, -1, idx)
    d = peak / -8.0
    inv = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    qs = torch.trunc((blocks * inv) + 8.5).clamp(0, 15).to(torch.uint8)
    qs = qs.reshape((n_blocks, 2, blocks.shape[-1] // 2))
    packed = (qs[..., 0, :] & 0x0F) | (qs[..., 1, :] << 4)
    header = d.to(parent.computation_dtype).view(torch.uint8)
    return torch.cat([header, packed], dim=-1)


def dequantize_torch(blocks: torch.Tensor, parent: Any) -> torch.Tensor:
    header = blocks[..., :2]
    payload = blocks[..., 2:]
    d = header.view(parent.computation_dtype)
    unpacked = quick_unpack_4bits(payload)
    return d * unpacked
