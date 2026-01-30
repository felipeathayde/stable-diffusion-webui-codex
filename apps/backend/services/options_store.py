"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: JSON-backed options store for backend and launchers.
Provides a small, typed facade over `apps/settings_values.json` so the API, runtime helpers, and launcher profile defaults can share a single
source of truth without importing legacy/compat shims. Includes per-component storage/compute dtype override keys (core/TE/VAE).

Symbols (top-level; keep in sync; no ghosts):
- `SETTINGS_PATH` (constant): Absolute path to `apps/settings_values.json` under the repo root.
- `load_values` (function): Reads the settings JSON from disk and returns a dict.
- `save_values` (function): Writes the settings JSON to disk (atomic overwrite).
- `get_value` (function): Reads a single option value with a fallback default.
- `set_values` (function): Persists a mapping of option updates and returns the updated keys.
- `OptionsSnapshot` (class): Typed snapshot of option values used by runtime/engines/launchers.
- `get_snapshot` (function): Builds an `OptionsSnapshot` from persisted values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from apps.backend.infra.config.repo_root import get_repo_root

SETTINGS_PATH = str(get_repo_root() / "apps" / "settings_values.json")


def load_values() -> Dict[str, Any]:
    if not os.path.exists(SETTINGS_PATH):
        return {}
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid settings file (expected object): {SETTINGS_PATH}")
    return data


def save_values(values: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(dict(values), f, indent=2)


def get_value(key: str, default: Any = None) -> Any:
    return load_values().get(key, default)


def set_values(payload: Mapping[str, Any]) -> list[str]:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    data = load_values()
    updated: list[str] = []
    for k, v in payload.items():
        key = str(k)
        data[key] = v
        updated.append(key)
    save_values(data)
    return updated


@dataclass
class OptionsSnapshot:
    codex_export_video: bool = False
    codex_core_device: str = "auto"
    codex_core_dtype: str = "auto"
    codex_core_compute_dtype: str = "auto"
    codex_te_device: str = "auto"
    codex_te_dtype: str = "auto"
    codex_te_compute_dtype: str = "auto"
    codex_vae_device: str = "auto"
    codex_vae_dtype: str = "auto"
    codex_vae_compute_dtype: str = "auto"
    codex_smart_offload: bool = False
    codex_smart_fallback: bool = False
    codex_smart_cache: bool = True
    codex_core_streaming: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "codex_export_video": self.codex_export_video,
            "codex_core_device": self.codex_core_device,
            "codex_core_dtype": self.codex_core_dtype,
            "codex_core_compute_dtype": self.codex_core_compute_dtype,
            "codex_te_device": self.codex_te_device,
            "codex_te_dtype": self.codex_te_dtype,
            "codex_te_compute_dtype": self.codex_te_compute_dtype,
            "codex_vae_device": self.codex_vae_device,
            "codex_vae_dtype": self.codex_vae_dtype,
            "codex_vae_compute_dtype": self.codex_vae_compute_dtype,
            "codex_smart_offload": self.codex_smart_offload,
            "codex_smart_fallback": self.codex_smart_fallback,
            "codex_smart_cache": self.codex_smart_cache,
            "codex_core_streaming": self.codex_core_streaming,
        }


def get_snapshot() -> OptionsSnapshot:
    v = load_values()

    def _str_value(key: str, default: str) -> str:
        raw = v.get(key)
        if raw is None:
            return default
        text = str(raw).strip()
        return text or default

    return OptionsSnapshot(
        codex_export_video=bool(v.get("codex_export_video", False)),
        codex_core_device=_str_value("codex_core_device", "auto"),
        codex_core_dtype=_str_value("codex_core_dtype", "auto"),
        codex_core_compute_dtype=_str_value("codex_core_compute_dtype", "auto"),
        codex_te_device=_str_value("codex_te_device", "auto"),
        codex_te_dtype=_str_value("codex_te_dtype", "auto"),
        codex_te_compute_dtype=_str_value("codex_te_compute_dtype", "auto"),
        codex_vae_device=_str_value("codex_vae_device", "auto"),
        codex_vae_dtype=_str_value("codex_vae_dtype", "auto"),
        codex_vae_compute_dtype=_str_value("codex_vae_compute_dtype", "auto"),
        codex_smart_offload=bool(v.get("codex_smart_offload", False)),
        codex_smart_fallback=bool(v.get("codex_smart_fallback", False)),
        codex_smart_cache=bool(v.get("codex_smart_cache", True)),
        codex_core_streaming=bool(v.get("codex_core_streaming", False)),
    )


__all__ = [
    "SETTINGS_PATH",
    "OptionsSnapshot",
    "get_snapshot",
    "get_value",
    "load_values",
    "save_values",
    "set_values",
]
