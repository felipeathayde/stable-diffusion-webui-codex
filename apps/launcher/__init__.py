from __future__ import annotations

"""Codex launcher public surface.

This package exposes the launcher infrastructure used by the BIOS/TUI and
other entrypoints.  Modules are organised for clarity:

- ``paths``: canonical path resolution helpers.
- ``log_buffer``: reusable in-memory log ring buffer.
- ``checks``: environment validation (Python, Node, Vite).
- ``services``: service supervision primitives.
- ``profiles``: segmented profile persistence (areas/models/meta).
"""

from .paths import CodexPaths, resolve_paths
from .log_buffer import CodexLogBuffer
from .checks import CodexLaunchCheck, run_launch_checks
from .services import (
    CodexServiceHandle,
    CodexServiceSpec,
    ServiceStatus,
    default_services,
)
from .profiles import (
    LauncherMeta,
    LauncherProfileStore,
)

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
]
