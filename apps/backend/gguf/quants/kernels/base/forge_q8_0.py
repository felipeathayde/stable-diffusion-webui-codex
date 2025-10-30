from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ...utils import np_roundf

__all__ = [
    "quantize_numpy",
    "dequantize_numpy",
    "quantize_torch",
    "dequantize_torch",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — Q8_0
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


def quantize_numpy(blocks: np.ndarray) -> np.ndarray:
    d = np.abs(blocks).max(axis=1, keepdims=True) / 127.0
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, 0, 1.0 / d)
    qs = np_roundf(blocks * inv)
    header = d.astype(np.float16).view(np.uint8)
    payload = qs.astype(np.int8).view(np.uint8)
    return np.concatenate([header, payload], axis=1)


def dequantize_numpy(blocks: np.ndarray) -> np.ndarray:
    header, payload = np.split(blocks, [2], axis=1)
    d = header.view(np.float16).astype(np.float32)
    x = payload.view(np.int8).astype(np.float32)
    return x * d


def quantize_torch(blocks: torch.Tensor, parent: Any) -> torch.Tensor:
    d = torch.abs(blocks).max(dim=1, keepdim=True).values / 127.0
    inv = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    qs = torch.round(blocks * inv)
    header = d.to(parent.computation_dtype).view(torch.int8)
    payload = qs.to(torch.int8)
    return torch.cat([header, payload], dim=1)


def dequantize_torch(blocks: torch.Tensor, parent: Any) -> torch.Tensor:
    header = blocks[..., :2]
    payload = blocks[..., 2:]
    d = header.view(parent.computation_dtype)
    x = payload.view(torch.int8).to(parent.computation_dtype)
    return x * d
