from __future__ import annotations

import numpy as np

from ....constants import GGML_QUANT_SIZES, GGMLQuantizationType

__all__ = [
    "KVALUES",
    "dequantize_numpy",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — IQ4_NL
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


KVALUES: tuple[int, ...] = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
BLOCK_SIZE = GGML_QUANT_SIZES[GGMLQuantizationType.IQ4_NL][0]


def dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]

    d, qs = np.hsplit(blocks, [2])

    d = d.view(np.float16).astype(np.float32)

    qs = qs.reshape((n_blocks, -1, 1, BLOCK_SIZE // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))

    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 1))

    kvalues = np.array(KVALUES, dtype=np.int8).reshape(1, 1, len(KVALUES))
    qs = np.take_along_axis(kvalues, qs, axis=-1).astype(np.float32).reshape((n_blocks, -1))

    return d * qs
