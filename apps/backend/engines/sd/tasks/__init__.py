"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Stable Diffusion task package marker (reserved).
Keeps task wiring under a stable namespace without importing SD engine/runtime modules at import time.

Symbols (top-level; keep in sync; no ghosts):
- `apps.backend.engines.sd.tasks` (module): Package marker for SD engine task wiring.
"""
