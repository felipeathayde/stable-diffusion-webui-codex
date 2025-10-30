from __future__ import annotations

import numpy as np

from ....constants import QK_K
from .forge_iq4_nl import KVALUES

__all__ = [
    "dequantize_numpy",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — IQ4_XS
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


def dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]

    d, rest = np.hsplit(blocks, [2])
    scales_h, rest = np.hsplit(rest, [2])
    scales_l, qs = np.hsplit(rest, [QK_K // 64])

    d = d.view(np.float16).astype(np.float32)
    scales_h = scales_h.view(np.uint16)

    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, 1, -1)) >> np.array([2 * i for i in range(QK_K // 32)], dtype=np.uint16).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & np.uint8(0x0F)
    scales_h = scales_h.reshape((n_blocks, -1)).astype(np.uint8) & np.uint8(0x03)

    scales = (scales_l | (scales_h << np.uint8(4))).astype(np.int8) - np.int8(32)
    dl = (d * scales.astype(np.float32)).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = qs.reshape((n_blocks, -1, 32, 1)) & np.uint8(0x0F)

    kvalues = np.array(KVALUES, dtype=np.int8).reshape((1, 1, 1, -1))
    qs = np.take_along_axis(kvalues, qs, axis=-1).astype(np.float32).reshape((n_blocks, -1, 32))

    return (dl * qs).reshape((n_blocks, QK_K))
