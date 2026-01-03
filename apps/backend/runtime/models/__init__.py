"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime models package facade with lazy exports for model loading, registries, and state-dict helpers.
This package must stay dependency-light at import time (API factories and tests may import it without a full torch install).

Symbols (top-level; keep in sync; no ghosts):
- `_EXPORTS` (constant): Mapping `{name: module_path}` for runtime models submodules exposed via `__getattr__`.
- `__getattr__` (function): Lazy import hook resolving runtime models submodules on first access.
- `__all__` (constant): List of exported runtime models submodule names (keys of `_EXPORTS`).
"""

from __future__ import annotations

_EXPORTS = {
    "api": "apps.backend.runtime.models.api",
    "loader": "apps.backend.runtime.models.loader",
    "registry": "apps.backend.runtime.models.registry",
    "safety": "apps.backend.runtime.models.safety",
    "state_dict": "apps.backend.runtime.models.state_dict",
    "types": "apps.backend.runtime.models.types",
}


def __getattr__(name: str):  # pragma: no cover - import-time laziness
    modpath = _EXPORTS.get(name)
    if not modpath:
        raise AttributeError(name)
    import importlib

    module = importlib.import_module(modpath)
    return module


__all__ = list(_EXPORTS.keys())
