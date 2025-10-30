from __future__ import annotations

import numpy as np

from ....constants import QK_K
from .forge_iq1_s import DELTA, GRID_HEX, GRID_MAP, GRID_SHAPE

__all__ = [
    "GRID_SHAPE",
    "GRID_MAP",
    "GRID_HEX",
    "DELTA",
    "dequantize_numpy",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — IQ1_M
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


def dequantize_numpy(blocks: np.ndarray, grid: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]

    qs, rest = np.hsplit(blocks, [QK_K // 8])
    qh, scales = np.hsplit(rest, [QK_K // 16])

    scales = scales.view(np.uint16)
    d = (scales.reshape((n_blocks, 4)) & np.uint16(0xF000)) >> np.array([12, 8, 4, 0], dtype=np.uint16).reshape((1, 4))
    d = d[..., 0] | d[..., 1] | d[..., 2] | d[..., 3]
    d = d.view(np.float16).astype(np.float32).reshape((n_blocks, 1))

    scales = scales.reshape(n_blocks, -1, 1) >> np.array([0, 3, 6, 9], dtype=np.uint16).reshape((1, 1, 4))
    scales = (scales & 0x07).reshape((n_blocks, -1))
    dl = d * (2 * scales + 1)
    dl = dl.reshape((n_blocks, -1, 2, 1, 1))

    qh = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
    qs = qs.astype(np.uint16) | ((qh & 0x07).astype(np.uint16) << 8).reshape((n_blocks, -1))

    delta = np.where(qh & 0x08 == 0, DELTA, -DELTA)
    delta = delta.reshape((n_blocks, -1, 2, 2, 1))

    grid = np.take_along_axis(grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
    grid = grid.reshape((n_blocks, -1, 2, 2, 8))

    return (dl * (grid + delta)).reshape((n_blocks, QK_K))
