from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ....constants import QK_K
from ....utils import quick_split

__all__ = [
    "dequantize_numpy",
    "dequantize_torch",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — Q6_K
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


def dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
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


def dequantize_torch(blocks: torch.Tensor, parameter: Any) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = quick_split(blocks, [QK_K // 2, QK_K // 4, QK_K // 16])
    scales = scales.view(torch.int8).to(parameter.computation_dtype)
    d = d.view(torch.float16).to(parameter.computation_dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(parameter.computation_dtype) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))
