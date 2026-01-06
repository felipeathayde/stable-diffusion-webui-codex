"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: JSON-backed options store for backend and launchers.
Provides a small, typed facade over `apps/settings_values.json` so the API, runtime helpers, and launcher profile defaults can share a single
source of truth without importing legacy/compat shims.

Symbols (top-level; keep in sync; no ghosts):
- `SETTINGS_PATH` (constant): Absolute path to `apps/settings_values.json` under the repo root.
- `load_values` (function): Reads the settings JSON from disk and returns a dict.
- `save_values` (function): Writes the settings JSON to disk (atomic overwrite).
- `get_value` (function): Reads a single option value with a fallback default.
- `set_values` (function): Persists a mapping of option updates and returns the updated keys.
- `OptionsSnapshot` (class): Typed snapshot of option values used by runtime/engines/launchers.
- `get_snapshot` (function): Builds an `OptionsSnapshot` from persisted values.
- `get_selected_vae` (function): Convenience accessor for the selected VAE label/path (`sd_vae`).
- `get_mode` (function): Convenience accessor for the current mode string (UI-facing).
- `get_engine` (function): Convenience accessor for the current engine key string (UI-facing).
- `get_current_checkpoint` (function): Convenience accessor for the configured checkpoint name/path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

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


def get_selected_vae(default: str = "Automatic") -> str:
    return str(get_value("sd_vae", default))


def get_mode(default: str = "Normal") -> str:
    return str(get_value("codex_mode", default))


def get_engine(default: str = "sd15") -> str:
    return str(get_value("codex_engine", default))


def get_current_checkpoint(default: str | None = None) -> str | None:
    v = get_value("sd_model_checkpoint", default)
    return None if v is None else str(v)


@dataclass
class OptionsSnapshot:
    codex_mode: str = "Normal"
    codex_engine: str = "sd15"
    sd_model_checkpoint: Optional[str] = None
    codex_export_video: bool = False
    sd_vae: str = "Automatic"
    codex_diffusion_device: Optional[str] = None
    codex_diffusion_dtype: Optional[str] = None
    codex_te_device: Optional[str] = None
    codex_te_dtype: Optional[str] = None
    codex_vae_device: Optional[str] = None
    codex_vae_dtype: Optional[str] = None
    codex_smart_offload: bool = False
    codex_smart_fallback: bool = False
    codex_smart_cache: bool = False
    codex_core_streaming: bool = False
    codex_wan22_use_spec_runtime: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "codex_mode": self.codex_mode,
            "codex_engine": self.codex_engine,
            "sd_model_checkpoint": self.sd_model_checkpoint,
            "codex_export_video": self.codex_export_video,
            "sd_vae": self.sd_vae,
            "codex_diffusion_device": self.codex_diffusion_device,
            "codex_diffusion_dtype": self.codex_diffusion_dtype,
            "codex_te_device": self.codex_te_device,
            "codex_te_dtype": self.codex_te_dtype,
            "codex_vae_device": self.codex_vae_device,
            "codex_vae_dtype": self.codex_vae_dtype,
            "codex_smart_offload": self.codex_smart_offload,
            "codex_smart_fallback": self.codex_smart_fallback,
            "codex_smart_cache": self.codex_smart_cache,
            "codex_core_streaming": self.codex_core_streaming,
            "codex_wan22_use_spec_runtime": self.codex_wan22_use_spec_runtime,
        }


def get_snapshot() -> OptionsSnapshot:
    v = load_values()
    return OptionsSnapshot(
        codex_mode=str(v.get("codex_mode", "Normal")),
        codex_engine=str(v.get("codex_engine", "sd15")),
        sd_model_checkpoint=v.get("sd_model_checkpoint"),
        codex_export_video=bool(v.get("codex_export_video", False)),
        sd_vae=str(v.get("sd_vae", "Automatic")),
        codex_diffusion_device=v.get("codex_diffusion_device"),
        codex_diffusion_dtype=v.get("codex_diffusion_dtype"),
        codex_te_device=v.get("codex_te_device"),
        codex_te_dtype=v.get("codex_te_dtype"),
        codex_vae_device=v.get("codex_vae_device"),
        codex_vae_dtype=v.get("codex_vae_dtype"),
        codex_smart_offload=bool(v.get("codex_smart_offload", False)),
        codex_smart_fallback=bool(v.get("codex_smart_fallback", False)),
        codex_smart_cache=bool(v.get("codex_smart_cache", False)),
        codex_core_streaming=bool(v.get("codex_core_streaming", False)),
        codex_wan22_use_spec_runtime=bool(v.get("codex_wan22_use_spec_runtime", False)),
    )


__all__ = [
    "SETTINGS_PATH",
    "OptionsSnapshot",
    "get_current_checkpoint",
    "get_engine",
    "get_mode",
    "get_selected_vae",
    "get_snapshot",
    "get_value",
    "load_values",
    "save_values",
    "set_values",
]
