"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Chroma engine package marker (import-light).
Avoids importing torch-heavy engine modules at package import time; import `apps.backend.engines.chroma.chroma` directly when you need the engine class.

Symbols (top-level; keep in sync; no ghosts):
- `apps.backend.engines.chroma` (module): Package marker for Chroma engine code.
"""
