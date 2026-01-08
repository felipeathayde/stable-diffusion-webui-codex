"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed configuration dataclasses for Chroma architecture and distilled guidance.

Symbols (top-level; keep in sync; no ghosts):
- `ChromaGuidanceConfig` (dataclass): Distilled-guidance modulation network config (validated in `__post_init__`).
- `ChromaArchitectureConfig` (dataclass): Transformer config (positional + guidance defaults, derived `patch_area`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from apps.backend.runtime.flux.config import FluxPositionalConfig


@dataclass(frozen=True, slots=True)
class ChromaGuidanceConfig:
    out_dim: int
    hidden_dim: int
    layers: int

    def __post_init__(self) -> None:
        if self.out_dim <= 0 or self.hidden_dim <= 0 or self.layers <= 0:
            raise ValueError("Guidance config values must be positive")


@dataclass(frozen=True, slots=True)
class ChromaArchitectureConfig:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    double_blocks: int
    single_blocks: int
    qkv_bias: bool = False
    positional: FluxPositionalConfig = field(
        default_factory=lambda: FluxPositionalConfig(patch_size=2, axes_dim=(16, 16, 16))
    )
    guidance: ChromaGuidanceConfig = field(
        default_factory=lambda: ChromaGuidanceConfig(out_dim=2048, hidden_dim=2048, layers=4)
    )

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.vec_in_dim <= 0 or self.context_in_dim <= 0:
            raise ValueError("Input dimensions must be positive")
        if self.hidden_size <= 0 or self.num_heads <= 0:
            raise ValueError("Hidden size/head count must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.double_blocks < 0 or self.single_blocks < 0:
            raise ValueError("block counts must be non-negative")
        if self.positional.positional_dim != self.hidden_size // self.num_heads:
            raise ValueError("Positional dimension mismatch with attention heads")

    @property
    def patch_area(self) -> int:
        return self.positional.patch_size * self.positional.patch_size
