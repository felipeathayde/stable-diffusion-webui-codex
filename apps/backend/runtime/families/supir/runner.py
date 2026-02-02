"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR enhance runner (not yet ported).
This module will own the SUPIR preprocess, sampling, optional tiling, and VAE cache behavior.

Symbols (top-level; keep in sync; no ghosts):
- `run_supir_enhance` (function): Enhance a single RGB image via SUPIR (not yet ported).
"""

from __future__ import annotations

from typing import Any

from .config import SupirEnhanceConfig
from .loader import SupirResolvedAssets


def run_supir_enhance(image, *, config: SupirEnhanceConfig, assets: SupirResolvedAssets) -> Any:
    raise NotImplementedError("SUPIR runner not yet ported")


__all__ = ["run_supir_enhance"]
