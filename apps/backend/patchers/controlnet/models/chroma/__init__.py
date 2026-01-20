"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Removed ControlNet Chroma facade (no compat shims).
Import Chroma placeholders/implementations from `apps.backend.patchers.controlnet.architectures.chroma` instead.

Symbols (top-level; keep in sync; no ghosts):
- `__all__` (constant): Empty export list (module import is intentionally rejected).
"""

__all__: list[str] = []

raise ImportError(
    "apps.backend.patchers.controlnet.models.chroma has been removed.\n"
    "Use apps.backend.patchers.controlnet.architectures.chroma instead."
)
