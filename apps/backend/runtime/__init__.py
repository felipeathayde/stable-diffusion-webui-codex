"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime package facade with lazy exports for backend execution.
Avoids importing heavy dependencies at package import time by exposing runtime submodules via a `__getattr__` lazy import hook.

Symbols (top-level; keep in sync; no ghosts):
- `_EXPORTS` (constant): Mapping `{name: module_path}` for runtime submodules exposed via the lazy import hook.
- `__getattr__` (function): Lazy import hook resolving runtime submodules on first access.
- `__all__` (constant): List of exported runtime submodule names (keys of `_EXPORTS`).
"""

_EXPORTS = {
    # Core utilities / small helpers
    "utils": "apps.backend.runtime.utils",
    "trace": "apps.backend.runtime.diagnostics.trace",
    # Memory stack
    "memory_management": "apps.backend.runtime.memory.memory_management",
    "stream": "apps.backend.runtime.memory.stream",
    # Heavier modules that may rely on memory
    "attention": "apps.backend.runtime.attention",
    "errors": "apps.backend.runtime.errors",
    "logging": "apps.backend.runtime.logging",
    "models": "apps.backend.runtime.models",
    "nn": "apps.backend.runtime.nn",
    "ops": "apps.backend.runtime.ops",
    "processing": "apps.backend.runtime.processing",
    "text_processing": "apps.backend.runtime.text_processing",
}


def __getattr__(name: str):  # pragma: no cover - import-time laziness
    modpath = _EXPORTS.get(name)
    if not modpath:
        raise AttributeError(name)
    import importlib
    module = importlib.import_module(modpath)
    return module


__all__ = list(_EXPORTS.keys())
