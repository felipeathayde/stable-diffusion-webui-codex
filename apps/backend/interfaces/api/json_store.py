"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Small JSON load/save helpers for API persistence files.
Used by UI persistence endpoints (tabs/workflows/presets), settings schema, and paths options.

Symbols (top-level; keep in sync; no ghosts):
- `_load_json` (function): Loads JSON from disk, returning `{}` on missing/unreadable files.
- `_save_json` (function): Saves JSON to disk (best-effort).
"""

from __future__ import annotations

import json
import os


def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass
