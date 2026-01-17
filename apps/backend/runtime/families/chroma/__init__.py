"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public exports for the Chroma transformer runtime (model + typed configs).

Symbols (top-level; keep in sync; no ghosts):
- `ChromaArchitectureConfig` (dataclass): Architecture config for Chroma (dims, blocks, positional + guidance defaults).
- `ChromaGuidanceConfig` (dataclass): Distilled-guidance modulation config (out_dim/hidden_dim/layers).
- `ChromaTransformer2DModel` (class): Chroma transformer implementation (Flux-like 2D transformer + guidance modulation).
"""

from .config import ChromaArchitectureConfig, ChromaGuidanceConfig
from .chroma import ChromaTransformer2DModel

__all__ = [
    "ChromaArchitectureConfig",
    "ChromaGuidanceConfig",
    "ChromaTransformer2DModel",
]
