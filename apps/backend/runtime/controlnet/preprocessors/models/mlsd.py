"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: MLSD line segment detector model for ControlNet preprocessing.
Defines the MLSD network and utilities to decode line segments from its output maps.

Symbols (top-level; keep in sync; no ghosts):
- `MLSDConfig` (dataclass): Weights/input-size configuration for the MLSD model.
- `MLSD` (class): MLSD network wrapper.
- `load_mlsd_model` (function): Loads and validates the MLSD model weights.
- `decode_lines` (function): Decodes line segments from the MLSD tensor prediction map.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_state_dict, resolve_weights_file


@dataclass(frozen=True)
class MLSDConfig:
    weights_path: str = "mlsd/mlsd_large_512_fp32.pth"
    input_size: Tuple[int, int] = (512, 512)


class BlockTypeA(nn.Module):
    def __init__(self, in_c1: int, in_c2: int, out_c1: int, out_c2: int, upscale: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c2, out_c2, kernel_size=1),
            nn.BatchNorm2d(out_c2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, kernel_size=1),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True),
        )
        self.upscale = upscale

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        b = self.conv1(b)
        a = self.conv2(a)
        if self.upscale:
            b = F.interpolate(b, scale_factor=2.0, mode="bilinear", align_corners=True)
        return torch.cat((a, b), dim=1)


class BlockTypeB(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) + x
        return self.conv2(x)


class BlockTypeC(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, groups: int = 1) -> None:
        if stride == 2:
            padding = 0
        else:
            padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        ]
        super().__init__(*layers)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        return super().forward(x)


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_channel = 32
        width_mult = 1.0
        round_nearest = 8
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        features = [ConvBNReLU(4, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features = nn.Sequential(*features)

        self.block_a1 = BlockTypeA(24, 32, 64, 96)
        self.block_a2 = BlockTypeA(40, 96, 128, 192)
        self.block_a3 = BlockTypeA(80, 192, 256, 320)

        self.block_b1 = BlockTypeB(64, 64)
        self.block_b2 = BlockTypeB(128, 128)
        self.block_b3 = BlockTypeB(256, 256)
        self.block_b4 = BlockTypeB(512, 512)

        self.block_c = BlockTypeC(960, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        for layer in self.features:
            x = layer(x)
            feats.append(x)

        f2 = feats[3]
        f3 = feats[6]
        f4 = feats[13]
        f5 = feats[17]

        up1 = self.block_a1(f3, f4)
        up2 = self.block_a2(f2, up1)
        up3 = self.block_a3(feats[0], up2)

        b1 = self.block_b1(up3)
        b2 = self.block_b2(up2)
        b3 = self.block_b3(up1)
        b4 = self.block_b4(f5)

        concat = torch.cat([b1, b2, b3, b4], dim=1)
        return self.block_c(concat)


class MLSD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = MobileNetV2Backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


@lru_cache(maxsize=1)
def _cached_mlsd_state(weights_path: str) -> Dict[str, torch.Tensor]:
    path = resolve_weights_file(weights_path)
    state = load_state_dict(path)
    cleaned = {}
    for key, tensor in state.items():
        cleaned[key.replace("module.", "")] = tensor.detach().clone()
    return cleaned


def load_mlsd_model(config: MLSDConfig) -> MLSD:
    model = MLSD()
    state = _cached_mlsd_state(config.weights_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"MLSD weights missing parameters: {missing}")
    if unexpected:
        raise RuntimeError(f"MLSD weights contain unexpected parameters: {unexpected}")
    model.eval()
    return model


def decode_lines(tp_map: torch.Tensor, topk: int = 200, kernel_size: int = 3,
                 score_threshold: float = 0.1, dist_threshold: float = 20.0) -> Tuple[torch.Tensor, torch.Tensor]:
    b, c, h, w = tp_map.shape
    assert b == 1
    displacement = tp_map[:, 1:5].squeeze(0)
    center = tp_map[:, 0]
    heat = torch.sigmoid(center)
    hmax = F.max_pool2d(heat, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
    keep = (hmax == heat).float()
    heat = heat * keep
    heat = heat.view(-1)
    scores, indices = torch.topk(heat, topk)
    yy = torch.div(indices, w, rounding_mode="floor").unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    ptss = torch.cat((yy, xx), dim=-1)

    displacement = displacement.permute(1, 2, 0)
    displacement = displacement.reshape(h * w, 4)
    selected = displacement[indices]
    dist = torch.sqrt((selected[:, 0] - selected[:, 2]) ** 2 + (selected[:, 1] - selected[:, 3]) ** 2)

    mask = (scores > score_threshold) & (dist > dist_threshold)
    ptss = ptss[mask]
    scores = scores[mask]
    selected = selected[mask]

    lines = torch.stack(
        [
            ptss[:, 1] + selected[:, 0],
            ptss[:, 0] + selected[:, 1],
            ptss[:, 1] + selected[:, 2],
            ptss[:, 0] + selected[:, 3],
        ],
        dim=-1,
    )
    return lines, scores


def render_lines(lines: torch.Tensor, height: int, width: int, *, thickness: int = 1) -> torch.Tensor:
    if lines.numel() == 0:
        return torch.zeros(1, 1, height, width, dtype=torch.float32, device=lines.device)
    grid = torch.zeros(1, 1, height, width, dtype=torch.float32, device=lines.device)
    for line in lines:
        x0, y0, x1, y1 = line
        num = int(max(abs(x1 - x0), abs(y1 - y0)).item()) + 1
        ts = torch.linspace(0, 1, num, device=lines.device)
        xs = torch.round(x0 + (x1 - x0) * ts).long().clamp_(0, width - 1)
        ys = torch.round(y0 + (y1 - y0) * ts).long().clamp_(0, height - 1)
        grid[0, 0, ys, xs] = 1.0
        if thickness > 1:
            for offset in range(1, thickness):
                grid[0, 0, (ys + offset).clamp(0, height - 1), xs] = 1.0
                grid[0, 0, (ys - offset).clamp(0, height - 1), xs] = 1.0
                grid[0, 0, ys, (xs + offset).clamp(0, width - 1)] = 1.0
                grid[0, 0, ys, (xs - offset).clamp(0, width - 1)] = 1.0
    return grid
