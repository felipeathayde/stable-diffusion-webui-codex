"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vision runtime package facade exposing encoder subpackages.
Provides thin, dependency-light imports for vision helpers (CLIP vision specs, preprocessing, and encoder wrappers).

Symbols (top-level; keep in sync; no ghosts):
- `clip` (module): Codex-native CLIP vision runtime subpackage (specs/state-dict tooling/preprocessing/encoder wrapper).
- `__all__` (constant): Public export list for the package facade.
"""

from . import clip

__all__ = ["clip"]
