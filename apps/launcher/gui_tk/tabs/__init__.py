"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tk launcher tab implementations.
Each tab is implemented as a small class with a `frame` and `reload/refresh` hooks, keeping `app.py` readable (including the Manual Env Vars editor tab).

Symbols (top-level; keep in sync; no ghosts):
- `__all__` (constant): Explicit export list for tab classes.
"""

from __future__ import annotations

from .services import ServicesTab
from .runtime import RuntimeTab
from .manual_env_vars import ManualEnvVarsTab
from .diagnostics import DiagnosticsTab
from .logs import LogsTab

__all__ = [
    "ServicesTab",
    "RuntimeTab",
    "ManualEnvVarsTab",
    "DiagnosticsTab",
    "LogsTab",
]
