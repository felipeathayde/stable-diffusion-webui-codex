"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux engine package marker (import-light).
Avoids importing torch-heavy engine modules at package import time; import engine/task modules directly when needed.

Symbols (top-level; keep in sync; no ghosts):
- `apps.backend.engines.flux` (module): Package marker for Flux engine code.
"""
