"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend settings JSON read/write service.
Persists runtime settings in `apps/settings_values.json` relative to `CODEX_ROOT` so option access does not depend on the process CWD.

Symbols (top-level; keep in sync; no ghosts):
- `SETTINGS_PATH` (constant): Absolute path to `apps/settings_values.json` under the repo root.
- `OptionsService` (class): Reads/writes the settings JSON and exposes API-friendly accessors.
"""

from __future__ import annotations

from typing import Any, Dict

from .options_store import SETTINGS_PATH, load_values, save_values


class OptionsService:
    """Native options/config service (no legacy WebUI dependency).

    Reads/writes a JSON settings file under apps/settings_values.json.
    """

    def get_config(self) -> Dict[str, Any]:
        return load_values()

    def set_config(self, req: Dict[str, Any]) -> bool:
        if not isinstance(req, dict):
            raise TypeError("options payload must be a dict")
        save_values(req or {})
        return True

    def get_cmd_flags(self) -> Dict[str, Any]:
        # Not available nativamente; retornar vazio ou consolidar flags próprias
        return {}
