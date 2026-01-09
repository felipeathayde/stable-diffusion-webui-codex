"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Codex-native processing primitives describing a generation run (txt2img/img2img/video) without legacy `modules.*` wrappers.
This package intentionally avoids `__init__.py` facades; import types from `models.py` / `datatypes.py`.

Symbols (top-level; keep in sync; no ghosts):
- `__all__` (constant): Empty export list; import processing types from submodules.
"""

__all__ = []
