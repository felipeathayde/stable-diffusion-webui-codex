"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LeReS depth model for ControlNet preprocessing.
Defines a ResNeXt-based encoder/decoder and a loader that maps legacy state dict keys into the Codex module layout.

Symbols (top-level; keep in sync; no ghosts):
- `LeReSConfig` (dataclass): LeReS weights/device/dtype configuration.
- `RelDepthModel` (class): Depth model wrapper combining the encoder and decoder.
- `load_leres_model` (function): Loads the LeReS model and applies legacy state dict mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torchvision.models import resnext101_32x8d

from apps.backend.runtime.models.safety import safe_torch_load


@dataclass(frozen=True)
class LeReSConfig:
    """Configuration for LeReS depth model loading."""

    weights_path: str = "leres/res101.pth"
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32


class ResidualRefineBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.input = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        self.branch = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
        )
        self.activation = nn.ReLU(inplace=True)
        self._init_params()

    def _init_params(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.input(x)
        refined = primary + self.branch(primary)
        return self.activation(refined)


class FusionUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, scale_factor: int) -> None:
        super().__init__()
        self.pre = ResidualRefineBlock(in_channels, hidden_channels)
        self.post = ResidualRefineBlock(hidden_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        fused = self.pre(low)
        fused = fused + high
        refined = self.post(fused)
        return self.upsample(refined)


class OutputAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1, bias=True),
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True),
        )
        self._init_params()

    def _init_params(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResNeXtEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnext101_32x8d(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> Iterable[torch.Tensor]:
        x = self.stem(x)
        x = self.pool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4


class LeReSDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4 = 256, 512, 1024, 2048
        m1, m2, m3, m4 = 256, 256, 256, 512
        self.bridge = ResidualRefineBlock(c4, m4)
        self.conv32 = nn.Conv2d(m4, m3, kernel_size=3, padding=1, bias=True)
        self.up32 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.fuse16 = FusionUpsampleBlock(c3, m3, m3, scale_factor=2)
        self.fuse8 = FusionUpsampleBlock(c2, m2, m2, scale_factor=2)
        self.fuse4 = FusionUpsampleBlock(c1, m1, m1, scale_factor=2)
        self.output = OutputAdapter(m1, 1, scale_factor=2)

    def forward(self, features: Iterable[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = features
        x32 = self.bridge(f4)
        x32 = self.conv32(x32)
        x16 = self.up32(x32)
        x8 = self.fuse16(f3, x16)
        x4 = self.fuse8(f2, x8)
        x2 = self.fuse4(f1, x4)
        return self.output(x2)


class RelDepthModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = ResNeXtEncoder()
        self.decoder = LeReSDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@lru_cache(maxsize=1)
def load_leres_model(config: LeReSConfig) -> RelDepthModel:
    model = RelDepthModel()
    try:
        state = safe_torch_load(config.weights_path, map_location="cpu")
        if isinstance(state, dict) and "depth_model" in state:
            converted = _map_legacy_state_dict(state["depth_model"])
        else:
            converted = state
        model.load_state_dict(converted, strict=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"LeReS weights not found at {config.weights_path}."
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load LeReS weights; ensure the legacy state dict mapping is up to date."
        ) from exc

    device = config.device or torch.device("cpu")
    model.to(device=device, dtype=config.dtype)
    model.eval()
    return model


def _map_legacy_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mapped: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.startswith("encoder_modules.encoder."):
            new_key = key.replace("encoder_modules.encoder", "encoder")
        elif key.startswith("decoder_modules."):
            new_key = key.replace("decoder_modules", "decoder")
        else:
            continue
        mapped[new_key] = tensor
    return mapped
