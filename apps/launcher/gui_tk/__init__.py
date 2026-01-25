"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tk GUI launcher package for Codex.
Holds the modular Tk/ttk implementation (app/controller/tabs) used by the stable entrypoint `apps/codex_launcher.py`.

Symbols (top-level; keep in sync; no ghosts):
- `CodexLauncherApp` (class): Tk app class (re-export).
- `main` (function): GUI entrypoint (re-export).
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

from .app import CodexLauncherApp, main

__all__ = [
    "CodexLauncherApp",
    "main",
]

