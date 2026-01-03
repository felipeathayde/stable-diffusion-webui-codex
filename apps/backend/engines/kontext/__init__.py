"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Kontext engine facade.
Re-exports `Kontext`, a Flux-derived image-conditioned engine, for compatibility imports.

Symbols (top-level; keep in sync; no ghosts):
- `Kontext` (class): Flux.1 Kontext engine (re-export).
- `__all__` (constant): Explicit export list for the facade.
"""

from __future__ import annotations

from .kontext import Kontext

__all__ = ["Kontext"]
