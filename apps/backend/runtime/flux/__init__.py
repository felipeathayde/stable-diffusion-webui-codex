"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public exports for the Flux transformer runtime (model + typed configs).

Symbols (top-level; keep in sync; no ghosts):
- `FluxArchitectureConfig` (dataclass): Transformer architecture config (dims/blocks/positional/guidance).
- `FluxGuidanceConfig` (dataclass): Optional guidance embedding config.
- `FluxPositionalConfig` (dataclass): Positional embedding config for Flux-like DiT models.
- `FluxTransformer2DModel` (class): Codex-native Flux transformer implementation.
"""

from .config import FluxArchitectureConfig, FluxGuidanceConfig, FluxPositionalConfig
from .model import FluxTransformer2DModel

__all__ = [
    "FluxArchitectureConfig",
    "FluxGuidanceConfig",
    "FluxPositionalConfig",
    "FluxTransformer2DModel",
]
