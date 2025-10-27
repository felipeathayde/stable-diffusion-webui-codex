from __future__ import annotations

"""WAN latent normalization helpers (ComfyUI-inspired).

Provides simple per-channel mean/std normalization for WAN latent formats.
Defaults to Wan21 (16ch). No SD15 fallback here by design.
"""

from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class LatentNorm:
    name: str
    channels: int

    def process_in(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def process_out(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Wan21Norm(LatentNorm):
    """Per-channel mean/std normalization for 16-channel WAN latent (Comfy Wan21)."""

    def __init__(self) -> None:
        super().__init__(name='wan21', channels=16)
        # Values from ComfyUI comfy/latent_formats.py (Wan21)
        self._mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ], dtype=torch.float32).view(1, self.channels, 1, 1, 1)
        self._std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
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


class IdentityNorm(LatentNorm):
    def __init__(self, channels: int) -> None:
        super().__init__(name='none', channels=channels)


def resolve_norm(kind: str | None, channels: int) -> LatentNorm:
    k = (kind or 'wan21').strip().lower()
    if k in ('wan21', 'wan'):
        return Wan21Norm()
    if k in ('none', 'identity'):
        return IdentityNorm(channels)
    # fallback: wan21
    return Wan21Norm()
