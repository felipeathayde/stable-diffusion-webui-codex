from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = [
    "quantize_numpy",
    "dequantize_numpy",
    "quantize_torch",
    "dequantize_torch",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — Q5_0
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


def quantize_numpy(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    imax = np.abs(blocks).argmax(axis=-1, keepdims=True)
    max_vals = np.take_along_axis(blocks, imax, axis=-1)
    d = max_vals / -16.0
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, 0, 1.0 / d)
    q = np.trunc((np.float64(blocks) * np.float64(inv)) + np.float64(16.5)).astype(np.uint8)
    q = np.clip(q, 0, 31)
    qs = q.reshape((n_blocks, 2, blocks.shape[-1] // 2))
    qs = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))
    qh = np.packbits(q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=-1, bitorder="little").reshape(n_blocks, 4)
    header = d.astype(np.float16).view(np.uint8)
    return np.concatenate([header, qh, qs], axis=-1)


def dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
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


def _view_as_uint32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8).view(torch.uint32)


def quantize_torch(blocks: torch.Tensor, parent: Any) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    idx = torch.abs(blocks).argmax(dim=-1, keepdim=True)
    peak = torch.gather(blocks, -1, idx)
    d = peak / -16.0
    inv = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    q = torch.trunc((blocks.float() * inv.float()) + 16.5).clamp(0, 31).to(torch.uint8)
    qs = q.view(n_blocks, 2, blocks.shape[-1] // 2)
    qs_packed = (qs[..., 0, :] & 0x0F) | (qs[..., 1, :] << 4)
    qh_src = q.view(n_blocks, 32)
    qh_packed = torch.zeros((n_blocks, 4), dtype=torch.uint8, device=q.device)
    for i in range(4):
        base = i * 8
        qh_packed[:, i] = (
            (qh_src[:, base + 0] >> 4)
            | ((qh_src[:, base + 1] >> 3) & 0x02)
            | ((qh_src[:, base + 2] >> 2) & 0x04)
            | ((qh_src[:, base + 3] >> 1) & 0x08)
            | ((qh_src[:, base + 4] << 0) & 0x10)
            | ((qh_src[:, base + 5] << 1) & 0x20)
            | ((qh_src[:, base + 6] << 2) & 0x40)
            | ((qh_src[:, base + 7] << 3) & 0x80)
        )
    header = d.to(parent.computation_dtype).view(torch.uint8)
    return torch.cat([header, qh_packed, qs_packed], dim=-1)


def dequantize_torch(blocks: torch.Tensor, parent: Any) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, rest = torch.split(blocks, [2], dim=-1)
    qh, qs = torch.split(rest, [4], dim=-1)
    d = d.view(parent.computation_dtype)
    qh_uint32 = _view_as_uint32(qh)
    bit_positions = torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    high = ((qh_uint32.reshape((n_blocks, 1)) >> bit_positions) & 1).to(torch.uint8)
    low = (
        qs.reshape((n_blocks, -1, 1, qs.shape[-1]))
        >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    )
    low = (low & 0x0F).reshape((n_blocks, -1))
    values = (low | (high << 4)).to(torch.int8) - 16
    return d * values.to(parent.computation_dtype)
