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
# Codex Forge Port — Q3_K
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


def dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
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
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(3)
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(1))
    qh = qh ^ np.uint8(1)
    q = (ql.astype(np.int8) - (qh << np.uint8(2)).astype(np.int8)).astype(np.float32)

    return (dl * q).reshape((n_blocks, QK_K))


def dequantize_torch(blocks: torch.Tensor, parameter: Any) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = quick_split(blocks, [QK_K // 8, QK_K // 4, 12])
    d = d.view(torch.float16).to(parameter.computation_dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))
    return (dl * q).reshape((n_blocks, QK_K))
