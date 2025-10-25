from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from .devices import default_device


@dataclass
class ImageRNG:
    shape: Tuple[int, int, int]
    seeds: Sequence[int]
    subseeds: Sequence[int]
    subseed_strength: float
    seed_resize_from_h: int = 0
    seed_resize_from_w: int = 0

    def next(self) -> torch.Tensor:
        bs = len(self.seeds)
        g = torch.Generator(device=default_device())
        g.manual_seed(int(self.seeds[0]) if self.seeds else 0)
        return torch.randn((bs, *self.shape), generator=g, device=default_device())


__all__ = ["ImageRNG"]

