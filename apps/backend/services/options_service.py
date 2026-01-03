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

import json
import os
from typing import Any, Dict

from apps.backend.infra.config.repo_root import get_repo_root

SETTINGS_PATH = str(get_repo_root() / 'apps' / 'settings_values.json')


class OptionsService:
    """Native options/config service (no legacy WebUI dependency).

    Reads/writes a JSON settings file under apps/settings_values.json.
    """

    def get_config(self) -> Dict[str, Any]:
        try:
            with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def set_config(self, req: Dict[str, Any]) -> bool:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(req or {}, f, indent=2)
        return True

    def get_cmd_flags(self) -> Dict[str, Any]:
        # Not available nativamente; retornar vazio ou consolidar flags próprias
        return {}
