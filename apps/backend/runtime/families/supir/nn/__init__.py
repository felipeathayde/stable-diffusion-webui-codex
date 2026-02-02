"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR neural network modules (control + UNet adapters).
Contains the torch modules required for the SUPIR Enhance runtime. These are SUPIR-specific and should not be reused by other families.

Symbols (top-level; keep in sync; no ghosts):
- `GLVControl` (class): Control network that produces per-block control tensors for the SUPIR UNet.
- `LightGLVUNet` (class): SUPIR UNet variant that consumes control tensors and predicts noise.
"""

from __future__ import annotations

from .control import GLVControl
from .unet import LightGLVUNet

__all__ = ["GLVControl", "LightGLVUNet"]

