"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Video IO package marker (no facade exports).
Import IO helpers from their owning modules (e.g. `ffmpeg.py`) to keep dependencies explicit and avoid facade sprawl.

Symbols (top-level; keep in sync; no ghosts):
- `__all__` (constant): Explicit export list (intentionally empty; no re-exports).
"""

__all__: list[str] = []
