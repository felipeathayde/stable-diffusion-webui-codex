"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Zero-initialized SUPIR adapter modules.
Implements the small adapter modules used by the SUPIR UNet variant to fuse control tensors into the UNet feature stream.

These modules are intentionally named to remain weight-compatible with common SUPIR checkpoints.

Symbols (top-level; keep in sync; no ghosts):
- `normalization` (function): GroupNorm(32) helper.
- `zero_module` (function): Zero-initialize a module's parameters (in-place; returns module).
- `ZeroSFT` (class): Spatial Feature Transform adapter (gamma/beta) with a zero-initialized conditioning path.
- `ZeroCrossAttn` (class): Cross-attention adapter from control tensor → feature tensor (residual, scaled).
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from apps.backend.runtime.common.nn.unet.utils import conv_nd
from apps.backend.runtime.common.nn.unet.layers import CrossAttention


def normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        try:
            p.detach().zero_()
        except Exception:
            pass
    return module


class ZeroSFT(nn.Module):
    def __init__(self, label_nc: int, norm_nc: int, *, concat_channels: int = 0, norm: bool = True, mask: bool = False):
        super().__init__()

        ks = 3
        pw = ks // 2

        self.norm = bool(norm)
        self.param_free_norm = normalization(norm_nc + concat_channels) if self.norm else nn.Identity()

        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.SiLU(),
        )
        self.zero_mul = zero_module(nn.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw))
        self.zero_add = zero_module(nn.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw))

        self.zero_conv = zero_module(conv_nd(2, label_nc, norm_nc, 1, 1, 0))
        self.pre_concat = bool(concat_channels != 0)
        self.mask = bool(mask)

    @torch.inference_mode()
    def forward(self, c: torch.Tensor, h: torch.Tensor, h_ori: torch.Tensor | None = None, *, control_scale: float = 1.0) -> torch.Tensor:
        if self.mask:
            # Masking is used only for progressive ablations; keep behaviour explicit.
            h = h + self.zero_conv(c) * torch.zeros_like(h)
        else:
            h = h + self.zero_conv(c)

        if h_ori is not None and self.pre_concat:
            h_raw = torch.cat([h_ori, h], dim=1)
        else:
            h_raw = h

        if h_ori is not None and self.pre_concat:
            h = torch.cat([h_ori, h], dim=1)

        actv = self.mlp_shared(c)
        gamma = self.zero_mul(actv)
        beta = self.zero_add(actv)
        if self.mask:
            gamma = gamma * torch.zeros_like(gamma)
            beta = beta * torch.zeros_like(beta)

        h = self.param_free_norm(h) * (gamma + 1) + beta

        if h_ori is not None and not self.pre_concat:
            h = torch.cat([h_ori, h], dim=1)

        # Blend in-place: h = control_scale * h + (1 - control_scale) * h_raw
        s = float(control_scale)
        h.mul_(s)
        h.add_(h_raw, alpha=(1.0 - s))
        return h


class ZeroCrossAttn(nn.Module):
    def __init__(self, context_dim: int, query_dim: int, *, mask: bool = False):
        super().__init__()
        heads = max(1, int(query_dim // 64))
        self.attn = CrossAttention(query_dim=query_dim, context_dim=context_dim, heads=heads, dim_head=64)
        self.norm1 = normalization(query_dim)
        self.norm2 = normalization(context_dim)
        self.mask = bool(mask)

    @torch.inference_mode()
    def forward(self, context: torch.Tensor, x: torch.Tensor, *, control_scale: float = 1.0) -> torch.Tensor:
        x_in = x
        x = self.norm1(x)
        context = self.norm2(context)
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        context = rearrange(context, "b c h w -> b (h w) c").contiguous()
        x = self.attn(x, context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if self.mask:
            x = x * torch.zeros_like(x)
        x.mul_(float(control_scale)).add_(x_in)
        return x


__all__ = ["ZeroCrossAttn", "ZeroSFT", "normalization", "zero_module"]

