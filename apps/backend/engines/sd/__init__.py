"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Stable Diffusion engine package marker (import-light).
Engine implementations live in sibling modules (`sd15.py`, `sd20.py`, `sd35.py`, `sdxl.py`); import them directly when registering engines.

Symbols (top-level; keep in sync; no ghosts):
- `apps.backend.engines.sd` (module): Package marker for SD family engines.
"""
