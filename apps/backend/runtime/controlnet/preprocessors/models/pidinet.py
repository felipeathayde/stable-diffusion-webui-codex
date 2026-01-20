"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: PiDiNet edge detector model for ControlNet preprocessing.
Defines PiDiNet variants and a loader that validates weight compatibility.

Symbols (top-level; keep in sync; no ghosts):
- `PiDiNetConfig` (dataclass): PiDiNet weights/variant configuration.
- `PiDiNet` (class): PiDiNet network module.
- `load_pidinet_model` (function): Loads and validates PiDiNet model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_state_dict, resolve_weights_file


_NET_VARIANTS: Dict[str, List[str]] = {
    "baseline": ["cv"] * 16,
    "c-v15": ["cd"] + ["cv"] * 15,
    "a-v15": ["ad"] + ["cv"] * 15,
    "r-v15": ["rd"] + ["cv"] * 15,
    "cvvv4": ["cd", "cv", "cv", "cv", "cd", "cv", "cv", "cv", "cd", "cv", "cv", "cv", "cd", "cv", "cv", "cv"],
    "avvv4": ["ad", "cv", "cv", "cv", "ad", "cv", "cv", "cv", "ad", "cv", "cv", "cv", "ad", "cv", "cv", "cv"],
    "rvvv4": ["rd", "cv", "cv", "cv", "rd", "cv", "cv", "cv", "rd", "cv", "cv", "cv", "rd", "cv", "cv", "cv"],
    "carv4": ["cd", "ad", "rd", "cv", "cd", "ad", "rd", "cv", "cd", "ad", "rd", "cv", "cd", "ad", "rd", "cv"],
    "c16": ["cd"] * 16,
    "a16": ["ad"] * 16,
    "r16": ["rd"] * 16,
}


@dataclass(frozen=True)
class PiDiNetConfig:
    weights_path: str = "pidinet/table5_pidinet.pth"
    variant: str = "carv4"
    dilation: int = 24
    use_spatial_attention: bool = True


def _create_operation(op_type: str):
    if op_type == "cv":
        return F.conv2d

    if op_type == "cd":
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func

    if op_type == "ad":
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            shape = weights.shape
            weights_flat = weights.view(shape[0], shape[1], -1)
            reordered = weights_flat[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
            weights_conv = (weights_flat - reordered).view(shape)
            return F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return func

    if op_type == "rd":
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            device = x.device
            shape = weights.shape
            buffer = torch.zeros(shape[0], shape[1], 5, 5, device=device, dtype=weights.dtype)
            weights_flat = weights.view(shape[0], shape[1], -1)
            buffer.view(shape[0], shape[1], -1)[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights_flat[:, :, 1:]
            buffer.view(shape[0], shape[1], -1)[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights_flat[:, :, 1:]
            return F.conv2d(x, buffer, bias, stride=stride, padding=2 * dilation, dilation=dilation, groups=groups)
        return func

    raise ValueError(f"Unknown PiDiNet operation '{op_type}'")


class PDCConv(nn.Module):
    def __init__(self, op_type: str, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, dilation: int = 1, groups: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.op = _create_operation(op_type)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class PDCBlock(nn.Module):
    def __init__(self, op_type: str, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.stride = stride
        if stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.depth_conv = PDCConv(op_type, in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.stride > 1:
            identity = self.pool(identity)
            identity = self.shortcut(identity)
            x = self.pool(x)
        y = self.depth_conv(x)
        y = self.relu(y)
        y = self.point_conv(y)
        return y + identity


class CSAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False)
        nn.init.constant_(self.conv1.bias, 0.0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(x)
        y = self.conv1(y)
        y = self.conv2(y)
        return x * self.sigmoid(y)


class CDCM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.dilated = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False)
            for d in (5, 7, 9, 11)
        )
        nn.init.constant_(self.conv1.bias, 0.0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        x = self.conv1(x)
        return sum(conv(x) for conv in self.dilated)


class MapReduce(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PiDiNet(nn.Module):
    def __init__(self, config: PiDiNetConfig) -> None:
        super().__init__()
        variant = config.variant
        if variant not in _NET_VARIANTS:
            raise ValueError(f"Unsupported PiDiNet variant '{variant}'")
        ops = _NET_VARIANTS[variant]

        self.init_conv = PDCConv(ops[0], 3, 60, kernel_size=3, padding=1)

        self.block1_1 = PDCBlock(ops[1], 60, 60)
        self.block1_2 = PDCBlock(ops[2], 60, 60)
        self.block1_3 = PDCBlock(ops[3], 60, 60)

        self.block2_1 = PDCBlock(ops[4], 60, 120, stride=2)
        self.block2_2 = PDCBlock(ops[5], 120, 120)
        self.block2_3 = PDCBlock(ops[6], 120, 120)
        self.block2_4 = PDCBlock(ops[7], 120, 120)

        self.block3_1 = PDCBlock(ops[8], 120, 240, stride=2)
        self.block3_2 = PDCBlock(ops[9], 240, 240)
        self.block3_3 = PDCBlock(ops[10], 240, 240)
        self.block3_4 = PDCBlock(ops[11], 240, 240)

        self.block4_1 = PDCBlock(ops[12], 240, 240, stride=2)
        self.block4_2 = PDCBlock(ops[13], 240, 240)
        self.block4_3 = PDCBlock(ops[14], 240, 240)
        self.block4_4 = PDCBlock(ops[15], 240, 240)

        fuse_channels = [60, 120, 240, 240]
        self.use_spatial_attention = config.use_spatial_attention
        self.dilation = config.dilation

        self.attentions = nn.ModuleList()
        self.dilations = nn.ModuleList()
        self.reductions = nn.ModuleList()

        for channels in fuse_channels:
            if self.dilation is not None:
                self.dilations.append(CDCM(channels, self.dilation))
                reduce_in = self.dilation
            else:
                reduce_in = channels
            if self.use_spatial_attention:
                self.attentions.append(CSAM(reduce_in))
            else:
                self.attentions.append(nn.Identity())
            self.reductions.append(MapReduce(reduce_in))

        self.classifier = nn.Conv2d(4, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        _, _, H, W = x.shape

        x1 = self.block1_3(self.block1_2(self.block1_1(self.init_conv(x))))
        x2 = self.block2_4(self.block2_3(self.block2_2(self.block2_1(x1))))
        x3 = self.block3_4(self.block3_3(self.block3_2(self.block3_1(x2))))
        x4 = self.block4_4(self.block4_3(self.block4_2(self.block4_1(x3))))

        stages = [x1, x2, x3, x4]
        fused_maps: List[torch.Tensor] = []
        for i, feature in enumerate(stages):
            if self.dilation is not None:
                feature = self.dilations[i](feature)
            feature = self.attentions[i](feature)
            reduced = self.reductions[i](feature)
            reduced = F.interpolate(reduced, size=(H, W), mode="bilinear", align_corners=False)
            fused_maps.append(torch.sigmoid(reduced))

        fused = self.classifier(torch.cat(fused_maps, dim=1))
        fused = torch.sigmoid(fused)
        fused_maps.append(fused)
        return fused_maps


@lru_cache(maxsize=2)
def _cached_pidinet_state(weights_path: str):
    path = resolve_weights_file(weights_path)
    state = load_state_dict(path)
    cleaned = {}
    for key, tensor in state.items():
        cleaned[key.replace("module.", "")] = tensor.detach().clone()
    return cleaned


def load_pidinet_model(config: PiDiNetConfig) -> PiDiNet:
    model = PiDiNet(config)
    state = _cached_pidinet_state(config.weights_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"PiDiNet weights missing parameters: {missing}")
    if unexpected:
        raise RuntimeError(f"PiDiNet weights contain unexpected parameters: {unexpected}")
    model.eval()
    return model
