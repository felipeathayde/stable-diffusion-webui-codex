"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher package public facade.
Re-exports path resolution, log buffering, environment checks, service supervision, and profile persistence used by TUI/BIOS and other entrypoints.

Symbols (top-level; keep in sync; no ghosts):
- `CodexPaths` (dataclass): Resolved runtime path container (re-export).
- `resolve_paths` (function): Resolves canonical paths with strict normalization (re-export).
- `CodexLogBuffer` (dataclass): Thread-safe ring buffer for launcher logs (re-export).
- `CodexLaunchCheck` (dataclass): Structured launcher preflight check result (re-export).
- `run_launch_checks` (function): Runs launcher environment preflight checks (re-export).
- `CodexServiceHandle` (dataclass): Handle for a supervised service process (re-export).
- `CodexServiceSpec` (dataclass): Service definition used by the launcher supervisor (re-export).
- `ServiceStatus` (enum): Service lifecycle status values (re-export).
- `default_services` (function): Constructs the default launcher service set (re-export).
- `LauncherMeta` (dataclass): Launcher metadata persisted alongside profiles (re-export).
- `LauncherProfileStore` (class): Profile store for launcher settings and selections (re-export).
- `DEFAULT_PYTORCH_ALLOC_CONF` (constant): Default `PYTORCH_ALLOC_CONF` applied by launchers when unset.
- `__all__` (constant): Explicit export list for this facade.
"""

from __future__ import annotations

from .paths import CodexPaths, resolve_paths
from .log_buffer import CodexLogBuffer
from .checks import CodexLaunchCheck, run_launch_checks
from .services import (
    CodexServiceHandle,
    CodexServiceSpec,
    ServiceStatus,
    default_services,
)
from .profiles import DEFAULT_PYTORCH_ALLOC_CONF, LauncherMeta, LauncherProfileStore

__all__ = [
    "CodexPaths",
    "resolve_paths",
    "CodexLogBuffer",
    "CodexLaunchCheck",
    "run_launch_checks",
    "CodexServiceHandle",
    "CodexServiceSpec",
    "ServiceStatus",
    "default_services",
    "LauncherMeta",
    "LauncherProfileStore",
    "DEFAULT_PYTORCH_ALLOC_CONF",
]
