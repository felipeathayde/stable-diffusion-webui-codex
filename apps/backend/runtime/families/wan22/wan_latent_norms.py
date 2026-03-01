"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN latent normalization helpers.
Provides per-channel mean/std normalization utilities for WAN latent formats (Wan21 16ch and Wan22 48ch).

Symbols (top-level; keep in sync; no ghosts):
- `WAN21_LATENTS_MEAN` (constant): Wan21 per-channel latent mean values (16ch).
- `WAN21_LATENTS_STD` (constant): Wan21 per-channel latent std values (16ch).
- `LatentNorm` (class): Base normalizer interface with `process_in`/`process_out` hooks.
- `Wan21Norm` (class): Per-channel normalization for 16-channel WAN latents.
- `Wan22Norm` (class): Per-channel normalization for 48-channel WAN latents.
- `IdentityNorm` (class): No-op normalization implementation for arbitrary channel counts.
- `resolve_norm` (function): Resolves the appropriate normalizer based on requested kind and latent channel count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

WAN21_LATENTS_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
WAN21_LATENTS_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


@dataclass
class LatentNorm:
    name: str
    channels: int

    def process_in(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def process_out(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def process_in_(self, x: torch.Tensor) -> torch.Tensor:
        return self.process_in(x)

    def process_out_(self, x: torch.Tensor) -> torch.Tensor:
        return self.process_out(x)


class Wan21Norm(LatentNorm):
    """Per-channel mean/std normalization for 16-channel WAN latent (Wan21)."""

    def __init__(self) -> None:
        super().__init__(name='wan21', channels=16)
        # Values from the Wan21 latent format reference.
        self._mean = torch.tensor(WAN21_LATENTS_MEAN, dtype=torch.float32).view(1, self.channels, 1, 1, 1)
        self._std = torch.tensor(WAN21_LATENTS_STD, dtype=torch.float32).view(1, self.channels, 1, 1, 1)

    def _as(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._mean.to(device=x.device, dtype=x.dtype),
            self._std.to(device=x.device, dtype=x.dtype),
        )

    def process_in(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._as(x)
        return (x - m) / s

    def process_out(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._as(x)
        return x * s + m

    def process_in_(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._as(x)
        x.sub_(m)
        x.div_(s)
        return x

    def process_out_(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._as(x)
        x.mul_(s)
        x.add_(m)
        return x


class IdentityNorm(LatentNorm):
    def __init__(self, channels: int) -> None:
        super().__init__(name='none', channels=channels)


class Wan22Norm(LatentNorm):
    """Per-channel mean/std normalization for 48-channel WAN latent (Wan22 T2V)."""

    def __init__(self) -> None:
        super().__init__(name='wan22', channels=48)
        # Values sourced from Wan22 latent statistics (reference values).
        self._mean = torch.tensor([
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
            -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
            -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
            -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
            -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
            0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
        ], dtype=torch.float32).view(1, self.channels, 1, 1, 1)
        self._std = torch.tensor([
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
            0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
            0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
            0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
            0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
        ], dtype=torch.float32).view(1, self.channels, 1, 1, 1)

    def _as(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._mean.to(device=x.device, dtype=x.dtype),
            self._std.to(device=x.device, dtype=x.dtype),
        )

    def process_in(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._as(x)
        return (x - m) / s

    def process_out(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._as(x)
        return x * s + m

    def process_in_(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._as(x)
        x.sub_(m)
        x.div_(s)
        return x

    def process_out_(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._as(x)
        x.mul_(s)
        x.add_(m)
        return x


def resolve_norm(kind: str | None, channels: int) -> LatentNorm:
    k = (kind or 'wan21').strip().lower()
    if k in ('wan22', 'wan2.2') or channels == 48:
        return Wan22Norm()
    if k in ('wan21', 'wan', 'wan2.1') or channels == 16:
        return Wan21Norm()
    if k in ('none', 'identity'):
        return IdentityNorm(channels)
    # fallback: match by channel count, default to Wan21
    if channels == 48:
        return Wan22Norm()
    if channels == 16:
        return Wan21Norm()
    return IdentityNorm(channels)
