"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux.1 family engine package marker (import-light).
Avoids importing torch-heavy engine modules at package import time; import engine modules directly when registering engines.

Symbols (top-level; keep in sync; no ghosts):
- `apps.backend.engines.flux` (module): Package marker for Flux.1 family engine code (`flux.py`, `chroma.py`, `kontext.py`).
"""
