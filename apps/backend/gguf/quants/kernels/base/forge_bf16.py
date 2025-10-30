from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "quantize_numpy",
    "dequantize_numpy",
    "quantize_torch",
    "dequantize_torch",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — BF16
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


def quantize_numpy(blocks: np.ndarray) -> np.ndarray:
    view = blocks.view(np.uint32)
    quiet = np.where(
        (view & 0x7FFF_FFFF) > 0x7F80_0000,
        (view & np.uint32(0xFFFF_0000)) | np.uint32(64 << 16),
        view,
    )
    rounded = (np.uint64(quiet) + (0x7FFF + ((quiet >> 16) & 1))) >> 16
    return rounded.astype(np.uint16).view(np.uint8)


def dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    return (blocks.view(np.int16).astype(np.int32) << 16).view(np.float32)


def quantize_torch(blocks: torch.Tensor) -> torch.Tensor:
    view = blocks.view(torch.int32)
    quiet = torch.where(
        (view & 0x7FFF_FFFF) > 0x7F80_0000,
        (view & 0xFFFF_0000) | torch.tensor(64 << 16, dtype=torch.int32, device=view.device),
        view,
    )
    rounded = (quiet.to(torch.int64) + (0x7FFF + ((quiet >> 16) & 1))) >> 16
    return rounded.to(torch.int16).view(torch.uint8)


def dequantize_torch(blocks: torch.Tensor) -> torch.Tensor:
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)
