"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima engine package marker (no facade exports).
Import the engine/runtime assembly helpers from their defining modules (`anima.py`, `spec.py`) to keep dependencies explicit and avoid facade sprawl.

Symbols (top-level; keep in sync; no ghosts):
- `__all__` (constant): Explicit export list (intentionally empty; no re-exports).
"""

__all__: list[str] = []

