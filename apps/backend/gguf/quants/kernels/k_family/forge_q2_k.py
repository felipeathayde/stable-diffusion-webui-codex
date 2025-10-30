from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ....constants import QK_K
from ...utils import quick_split

__all__ = [
    "dequantize_numpy",
    "dequantize_torch",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — Q2_K
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


def dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
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


def dequantize_torch(blocks: torch.Tensor, parameter: Any) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = quick_split(blocks, [QK_K // 16, QK_K // 4, 2])
    d = d.view(torch.float16).to(parameter.computation_dtype)
    dmin = dmin.view(torch.float16).to(parameter.computation_dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16)).to(parameter.computation_dtype)
    return (dl * qs - ml).reshape((n_blocks, -1))
