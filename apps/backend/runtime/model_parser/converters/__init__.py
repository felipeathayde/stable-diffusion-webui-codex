"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Component conversion helpers for the Codex model parser.
Re-exports CLIP/T5 conversion helpers used by family-specific parser plans.

Symbols (top-level; keep in sync; no ghosts):
- `convert_clip` (function): Generic CLIP converter (alias-aware; OpenCLIP key normalization).
- `convert_sd15_clip` (function): SD1.5 CLIP-L converter (drops runtime-reconstructed heads).
- `convert_sd20_clip` (function): SD2.x CLIP-H converter (transposes projection when required).
- `convert_sdxl_clip_g` (function): SDXL CLIP-G converter (projection transposition; keeps logit_scale).
- `convert_sdxl_clip_l` (function): SDXL CLIP-L converter (drops runtime-reconstructed heads).
- `convert_t5_encoder` (function): T5 encoder converter (enforces stable layer-norm dtype).
- `convert_umt5_encoder` (function): UMT5 encoder converter (T5-compatible normalization).
- `convert_t5xxl_encoder` (function): T5-XXL encoder converter (T5-compatible normalization).
"""

from .clip import (
    convert_clip,
    convert_sd15_clip,
    convert_sd20_clip,
    convert_sdxl_clip_g,
    convert_sdxl_clip_l,
)
from .t5 import convert_t5_encoder, convert_umt5_encoder, convert_t5xxl_encoder

__all__ = [
    "convert_clip",
    "convert_sd15_clip",
    "convert_sd20_clip",
    "convert_sdxl_clip_g",
    "convert_sdxl_clip_l",
    "convert_t5_encoder",
    "convert_umt5_encoder",
    "convert_t5xxl_encoder",
]
