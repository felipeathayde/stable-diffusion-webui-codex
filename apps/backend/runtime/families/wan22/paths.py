"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 runtime path normalization helpers.
Centralizes Windows-drive path normalization for WSL/Linux runtime environments so config parsing and stage loading stay consistent.

Symbols (top-level; keep in sync; no ghosts):
- `normalize_win_path` (function): Normalizes Windows drive paths to WSL-style `/mnt/<drive>/...` paths when running on non-Windows.
"""

from __future__ import annotations

import os


def normalize_win_path(path: str) -> str:
    """Normalize `C:\\...` style paths into WSL `/mnt/c/...` when not on Windows."""
    if os.name == "nt":
        return path
    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        drive = path[0].lower()
        rest = path[2:].lstrip("\\/")
        return f"/mnt/{drive}/" + rest.replace("\\\\", "/").replace("\\", "/")
    return path


__all__ = ["normalize_win_path"]
