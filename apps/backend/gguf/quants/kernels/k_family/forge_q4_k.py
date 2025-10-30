from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch

from ....constants import QK_K
from ....quick_4bits_ops import change_4bits_order
from ...utils import quick_split

__all__ = [
    "get_scale_min_numpy",
    "get_scale_min_pytorch",
    "dequantize_numpy",
    "dequantize_torch",
    "bake_pytorch",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — Q4_K
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


K_SCALE_SIZE = 12


def get_scale_min_numpy(scales: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_blocks = scales.shape[0]
    scales = scales.view(np.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = np.split(scales, 3, axis=-2)
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return (sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8)))


def get_scale_min_pytorch(scales: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return (sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8)))


def dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, rest = np.hsplit(blocks, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, qs = np.hsplit(rest, [K_SCALE_SIZE])

    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)

    sc, mn = get_scale_min_numpy(scales)

    d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
    dm = (dmin * mn.astype(np.float32)).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 32)).astype(np.float32)

    return (d * qs - dm).reshape((n_blocks, QK_K))


def dequantize_torch(blocks: torch.Tensor, parameter: Any) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, rest = torch.split(blocks, [2], dim=-1)
    dmin, rest = torch.split(rest, [2], dim=-1)
    scales, qs = torch.split(rest, [K_SCALE_SIZE], dim=-1)
    d = d.view(torch.float16).to(parameter.computation_dtype)
    dmin = dmin.view(torch.float16).to(parameter.computation_dtype)
    sc, mn = get_scale_min_pytorch(scales)
    d = (d * sc.to(parameter.computation_dtype)).reshape((n_blocks, -1, 1))
    dm = (dmin * mn.to(parameter.computation_dtype)).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32)).to(parameter.computation_dtype)
    return (d * qs - dm).reshape((n_blocks, QK_K))


def bake_pytorch(parameter: Any) -> None:
    blocks = parameter.data
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = quick_split(blocks, [2, 2, K_SCALE_SIZE])
    d = d.view(torch.float16).to(parameter.computation_dtype)
    dmin = dmin.view(torch.float16).to(parameter.computation_dtype)
    sc, mn = get_scale_min_pytorch(scales)
    d = (d * sc.to(parameter.computation_dtype)).reshape((n_blocks, -1, 1))
    dm = (dmin * mn.to(parameter.computation_dtype)).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32))
    qs = change_4bits_order(qs)
    d_uint8 = d.view(torch.uint8).reshape((n_blocks, -1))
    dm_uint8 = dm.view(torch.uint8).reshape((n_blocks, -1))
    qs_uint8 = qs.view(torch.uint8)
    parameter.data = torch.cat([d_uint8, dm_uint8, qs_uint8], dim=-1).contiguous()
