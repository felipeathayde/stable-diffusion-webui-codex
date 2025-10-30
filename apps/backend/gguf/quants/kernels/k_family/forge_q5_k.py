from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ....constants import QK_K
from ....utils import quick_split
from .forge_q4_k import get_scale_min_numpy, get_scale_min_pytorch, K_SCALE_SIZE

__all__ = [
    "dequantize_numpy",
    "dequantize_torch",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — Q5_K
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


def dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, rest = np.hsplit(blocks, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, rest = np.hsplit(rest, [K_SCALE_SIZE])
    qh, qs = np.hsplit(rest, [QK_K // 8])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)
    sc, mn = get_scale_min_numpy(scales)
    d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
    dm = (dmin * mn.astype(np.float32)).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh_bits = qh.reshape((n_blocks, -1, 1, 32)) >> np.arange(8, dtype=np.uint8).reshape((1, 1, 8, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
    qh_bits = (qh_bits & np.uint8(1)).reshape((n_blocks, -1, 32))
    q = (ql | (qh_bits << np.uint8(4))).astype(np.float32)
    return (d * q - dm).reshape((n_blocks, QK_K))


def dequantize_torch(blocks: torch.Tensor, parameter: Any) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = quick_split(blocks, [2, 2, K_SCALE_SIZE, QK_K // 8])
    d = d.view(torch.float16).to(parameter.computation_dtype)
    dmin = dmin.view(torch.float16).to(parameter.computation_dtype)
    sc, mn = get_scale_min_pytorch(scales)
    d = (d * sc.to(parameter.computation_dtype)).reshape((n_blocks, -1, 1))
    dm = (dmin * mn.to(parameter.computation_dtype)).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh_bits = qh.reshape((n_blocks, -1, 1, 32)) >> torch.arange(8, device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh_bits = (qh_bits & 1).reshape((n_blocks, -1, 32))
    q = (ql | (qh_bits << 4)).to(parameter.computation_dtype)
    return (d * q - dm).reshape((n_blocks, QK_K))
