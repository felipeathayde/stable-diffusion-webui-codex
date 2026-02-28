"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Video upscaling package marker (no facade exports).
Import upscaling helpers from their owning modules (for example `seedvr2_cli.py`) to keep dependencies explicit and avoid facade sprawl.

Symbols (top-level; keep in sync; no ghosts):
- `__all__` (constant): Explicit export list (intentionally empty; no re-exports).
"""

__all__: list[str] = []
