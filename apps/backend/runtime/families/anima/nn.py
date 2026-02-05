"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Small NN building blocks shared by the Anima runtime.
Keeps core modules import-light and avoids cross-family imports for basic layers (e.g. RMSNorm).

Symbols (top-level; keep in sync; no ghosts):
- `RMSNorm` (class): RMSNorm with fp32 compute and dtype-preserving output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(f"RMSNorm expects a floating-point input tensor; got dtype={x.dtype}.")
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        out = normed * self.weight.float()
        return out.to(dtype=dtype)
