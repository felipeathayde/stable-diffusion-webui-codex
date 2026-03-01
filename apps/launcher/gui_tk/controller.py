"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Non-UI controller for the Tk launcher.
Wraps launcher infrastructure (profiles, services, logs) behind a small imperative API so tabs don’t reach into internals.
Builds service-scoped environments so API-only manual env overlays are applied only to API starts/restarts.

Symbols (top-level; keep in sync; no ghosts):
- `LauncherController` (class): Holds store/services/log_buffer and provides service + persistence helpers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

from apps.launcher.log_buffer import CodexLogBuffer
from apps.launcher.profiles import LauncherProfileStore
from apps.launcher.services import CodexServiceHandle


@dataclass(slots=True)
class LauncherController:
    codex_root: Path
    store: LauncherProfileStore
    log_buffer: CodexLogBuffer
    services: Dict[str, CodexServiceHandle]

    def build_env(self) -> dict[str, str]:
        return self.store.build_env()

    def build_env_for_service(self, name: str) -> dict[str, str]:
        env = self.store.build_env()
        if str(name or "").strip().upper() != "API":
            return env
        env.update(self.store.build_manual_api_env_overlay())
        return env

    @property
    def external_terminal_supported(self) -> bool:
        return os.name == "nt"

    def start_service(self, name: str, *, external_terminal: bool) -> None:
        env = self.build_env_for_service(name)
        self.services[name].start(env, external_terminal=external_terminal)

    def restart_service(self, name: str, *, external_terminal: bool) -> None:
        env = self.build_env_for_service(name)
        self.services[name].restart(env, external_terminal=external_terminal)

    def stop_service(self, name: str, *, wait: float = 10.0) -> None:
        self.services[name].stop(wait=wait)

    def kill_service(self, name: str, *, wait: float = 10.0) -> None:
        self.services[name].kill(wait=wait)

    def start_all(self, *, external_terminal: bool) -> None:
        for name in ("API", "UI"):
            self.start_service(name, external_terminal=external_terminal)

    def stop_all(self, *, wait: float = 10.0) -> None:
        for name in ("UI", "API"):
            self.stop_service(name, wait=wait)

    def persist_tab_index(self, tab_index: int) -> None:
        self.store.meta.tab_index = int(tab_index)
        self.store.save_meta()

    def persist_external_terminal(self, enabled: bool) -> None:
        self.store.meta.external_terminal = bool(enabled)
        self.store.save_meta()

    def persist_window_geometry(self, geometry: str) -> None:
        self.store.meta.window_geometry = str(geometry)
        self.store.save_meta()

    def persist_show_advanced_controls(self, enabled: bool) -> None:
        self.store.meta.show_advanced_controls = bool(enabled)
        self.store.save_meta()

    def save_settings(self) -> None:
        self.store.save()

    def reload_store(self) -> None:
        self.store = LauncherProfileStore.load(root=self.store.root)
