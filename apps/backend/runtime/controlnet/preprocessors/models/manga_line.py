"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Manga-line model for ControlNet preprocessing.
Defines a ResSkip network and a loader that validates weight compatibility.

Symbols (top-level; keep in sync; no ghosts):
- `MangaLineConfig` (dataclass): Weights configuration for the manga-line model.
- `ResSkip` (class): ResSkip network producing a 1-channel line map.
- `load_manga_line_model` (function): Loads and validates the manga-line model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn

from .utils import load_state_dict, resolve_weights_file


@dataclass(frozen=True)
class MangaLineConfig:
    weights_path: str = "manga_line/res_skip.pth"


class BNReLUConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, upsample: bool = False) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.BatchNorm2d(in_channels, eps=1e-3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        ]
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Shortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, upsample: bool = False) -> None:
        super().__init__()
        self.process = in_channels != out_channels or stride != 1 or upsample
        if self.process:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)]
            if upsample:
                layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.model = nn.Sequential(*layers)
        else:
            self.model = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.process:
            return self.model(x) + y
        return x + y


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = BNReLUConv(in_channels, out_channels, 3, stride=stride)
        self.conv2 = BNReLUConv(out_channels, out_channels, 3)
        self.shortcut = Shortcut(in_channels, out_channels, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return self.shortcut(x, out)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = BNReLUConv(in_channels, out_channels, 3, upsample=True)
        self.conv2 = BNReLUConv(out_channels, out_channels, 3)
        self.shortcut = Shortcut(in_channels, out_channels, upsample=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return self.shortcut(x, out)


class ResSkip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = nn.Sequential(BasicBlock(1, 24), BasicBlock(24, 24))
        self.block1 = nn.Sequential(*[BasicBlock(24 if i == 0 else 48, 48, stride=2 if i == 2 else 1) for i in range(3)])
        self.block2 = nn.Sequential(*[BasicBlock(48 if i == 0 else 96, 96, stride=2 if i == 4 else 1) for i in range(5)])
        self.block3 = nn.Sequential(*[BasicBlock(96 if i == 0 else 192, 192, stride=2 if i == 6 else 1) for i in range(7)])
        self.block4 = nn.Sequential(*[BasicBlock(192 if i == 0 else 384, 384, stride=2 if i == 11 else 1) for i in range(12)])

        self.block5 = nn.Sequential(*[UpsampleBlock(384 if i == 0 else 192, 192) for i in range(7)])
        self.res1 = Shortcut(192, 192)

        self.block6 = nn.Sequential(*[UpsampleBlock(192 if i == 0 else 96, 96) for i in range(5)])
        self.res2 = Shortcut(96, 96)

        self.block7 = nn.Sequential(*[UpsampleBlock(96 if i == 0 else 48, 48) for i in range(3)])
        self.res3 = Shortcut(48, 48)

        self.block8 = nn.Sequential(*[UpsampleBlock(48 if i == 0 else 24, 24) for i in range(2)])
        self.res4 = Shortcut(24, 24)

        self.block9 = nn.Sequential(BasicBlock(24, 16), BasicBlock(16, 16))
        self.final = BNReLUConv(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x5 = self.block5(x4)
        res1 = self.res1(x3, x5)

        x6 = self.block6(res1)
        res2 = self.res2(x2, x6)

        x7 = self.block7(res2)
        res3 = self.res3(x1, x7)

        x8 = self.block8(res3)
        res4 = self.res4(x0, x8)

        x9 = self.block9(res4)
        return torch.tanh(self.final(x9))


@lru_cache(maxsize=1)
def _cached_manga_state(weights_path: str):
    path = resolve_weights_file(weights_path)
    state = load_state_dict(path)
    cleaned = {}
    for key, value in state.items():
        cleaned[key.replace("module.", "")] = value.detach().clone()
    return cleaned


def load_manga_line_model(config: MangaLineConfig) -> nn.Module:
    model = ResSkip()
    state = _cached_manga_state(config.weights_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Manga line weights missing parameters: {missing}")
    if unexpected:
        raise RuntimeError(f"Manga line weights contain unexpected parameters: {unexpected}")
    model.eval()
    return model
