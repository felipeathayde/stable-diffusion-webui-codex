"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Philox-based random number generator compatible with CUDA `torch.randn` output.
Implements a deterministic Philox 4x32 generator + Box–Muller transform so CPU-only runs can reproduce CUDA noise for determinism-sensitive workflows.

Symbols (top-level; keep in sync; no ghosts):
- `_PHILOX_M` (constant): Philox round multiplier constants.
- `_PHILOX_W` (constant): Philox key schedule constants.
- `_split_uint64_to_uint32` (function): Views `(N,) uint64` as `(2, N) uint32` words.
- `_philox_round` (function): Single Philox 4x32 round (in-place counter transform).
- `_philox4x32` (function): Runs multiple Philox rounds and returns the transformed counter stream.
- `_box_muller` (function): Converts uniform uint32 streams into normal floats via Box–Muller.
- `PhiloxGenerator` (dataclass): Deterministic generator exposing `randn()` compatible with `torch.randn` shapes/devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch


_PHILOX_M = (0xD2511F53, 0xCD9E8D57)
_PHILOX_W = (0x9E3779B9, 0xBB67AE85)


def _split_uint64_to_uint32(values: np.ndarray) -> np.ndarray:
    """View helper turning (N,) uint64 into (2, N) uint32 array."""

    view = values.view(np.uint32)
    return view.reshape(-1, 2).transpose(1, 0)


def _philox_round(counter: np.ndarray, key: np.ndarray) -> None:
    """Single Philox 4x32 round."""

    v1 = _split_uint64_to_uint32(counter[0].astype(np.uint64) * _PHILOX_M[0])
    v2 = _split_uint64_to_uint32(counter[2].astype(np.uint64) * _PHILOX_M[1])

    counter[0] = v2[1] ^ counter[1] ^ key[0]
    counter[1] = v2[0]
    counter[2] = v1[1] ^ counter[3] ^ key[1]
    counter[3] = v1[0]


def _philox4x32(counter: np.ndarray, key: np.ndarray, rounds: int = 10) -> np.ndarray:
    for _ in range(rounds - 1):
        _philox_round(counter, key)
        key[0] = key[0] + _PHILOX_W[0]
        key[1] = key[1] + _PHILOX_W[1]

    _philox_round(counter, key)
    return counter


def _box_muller(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return normally distributed numbers via Box–Muller transform."""

    inv = np.float32(2.3283064e-10)
    inv_tau = np.float32(2.3283064e-10 * 6.2831855)

    uniform = u * inv + inv / 2.0
    angle = v * inv_tau + inv_tau / 2.0

    radius = np.sqrt(-2.0 * np.log(uniform))
    return (radius * np.sin(angle)).astype(np.float32)


@dataclass
class PhiloxGenerator:
    """Deterministic Philox RNG compatible with CUDA ``torch.randn`` output."""

    seed: int
    offset: int = 0

    def _prepare_counter(self, total: int) -> Tuple[np.ndarray, np.ndarray]:
        counter = np.zeros((4, total), dtype=np.uint32)
        counter[0] = self.offset
        counter[2] = np.arange(total, dtype=np.uint32)
        self.offset += 1

        key_values = np.full(total, np.uint64(self.seed), dtype=np.uint64)
        key = _split_uint64_to_uint32(key_values)
        return counter, key

    def randn(self, shape: Iterable[int], *, device: torch.device) -> torch.Tensor:
        count = int(np.prod(tuple(shape)))
        if count <= 0:
            return torch.empty(tuple(shape), device=device)

        counter, key = self._prepare_counter(count)
        stream = _philox4x32(counter, key)

        samples = _box_muller(stream[0], stream[1]).reshape(tuple(shape))
        tensor = torch.from_numpy(samples)
        return tensor.to(device)


__all__ = ["PhiloxGenerator"]
