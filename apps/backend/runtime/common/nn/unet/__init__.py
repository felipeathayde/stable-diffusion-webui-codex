"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public facade for the Codex-native UNet runtime.
Exposes the primary model/config types; callers should import layers/utils from their defining modules.

Symbols (top-level; keep in sync; no ghosts):
- `UNet2DConditionModel` (class): UNet with cross-attention conditioning used by SD-family pipelines.
- `UNetConfig` (dataclass): Typed config describing channel/block/attention layout for `UNet2DConditionModel`.
"""

from __future__ import annotations

from .config import UNetConfig
from .model import UNet2DConditionModel

__all__ = [
    "UNet2DConditionModel",
    "UNetConfig",
]
