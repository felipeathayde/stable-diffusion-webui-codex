"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Embedding modules for Flux variants (rotary frequencies + simple MLP projections).

Symbols (top-level; keep in sync; no ghosts):
- `EmbedND` (class): Builds N-D rotary frequency embeddings from integer positional IDs.
- `MLPEmbedder` (class): Two-layer MLP used for time/guidance/vector projections.
"""

from __future__ import annotations

import torch
from torch import nn

from .geometry import build_rotary_frequencies


class EmbedND(nn.Module):
    """N-dimensional sinusoidal embedding composed of rotary frequencies."""

    def __init__(self, dim: int, theta: int, axes_dim: tuple[int, ...]) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        embeddings = torch.cat(
            [build_rotary_frequencies(ids[..., i], self.axes_dim[i], self.theta) for i in range(ids.shape[-1])],
            dim=-3,
        )
        return embeddings.unsqueeze(1)


class MLPEmbedder(nn.Module):
    """Two-layer MLP embedder used for time/guidance/vector projections."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.activation(self.in_layer(x)))
