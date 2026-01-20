"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: HED edge detector model for ControlNet preprocessing.
Defines the HED network module and a loader that validates weight compatibility.

Symbols (top-level; keep in sync; no ghosts):
- `HEDConfig` (dataclass): HED weights configuration (path under the ControlNet cache root).
- `ControlNetHED` (class): HED network module producing edge projections.
- `load_hed_model` (function): Loads and validates the HED model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_state_dict, resolve_weights_file


@dataclass(frozen=True)
class HEDConfig:
    weights_path: str = "hed/ControlNetHED.pth"


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: int) -> None:
        super().__init__()
        convs = []
        convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        for _ in range(1, layers):
            convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.convs = nn.ModuleList(convs)
        self.projection = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, *, down_sampling: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if down_sampling:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        h = x
        for conv in self.convs:
            h = F.relu(conv(h), inplace=True)
        return h, self.projection(h)


class ControlNetHED(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.block1 = DoubleConvBlock(3, 64, 2)
        self.block2 = DoubleConvBlock(64, 128, 2)
        self.block3 = DoubleConvBlock(128, 256, 3)
        self.block4 = DoubleConvBlock(256, 512, 3)
        self.block5 = DoubleConvBlock(512, 512, 3)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = x - self.norm
        h, proj1 = self.block1(h)
        h, proj2 = self.block2(h, down_sampling=True)
        h, proj3 = self.block3(h, down_sampling=True)
        h, proj4 = self.block4(h, down_sampling=True)
        _, proj5 = self.block5(h, down_sampling=True)
        return [proj1, proj2, proj3, proj4, proj5]


@lru_cache(maxsize=4)
def _cached_state_dict(weights_path: str) -> Dict[str, torch.Tensor]:
    path = resolve_weights_file(weights_path)
    state = load_state_dict(path)
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value.detach().clone()
    return cleaned


def load_hed_model(config: HEDConfig) -> ControlNetHED:
    model = ControlNetHED()
    state = _cached_state_dict(config.weights_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"HED weights missing parameters: {missing}")
    if unexpected:
        raise RuntimeError(f"HED weights contain unexpected parameters: {unexpected}")
    model.eval()
    return model
