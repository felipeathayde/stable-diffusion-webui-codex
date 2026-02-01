"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime ops facade exposing operational helpers (GGUF/swap) via lazy attribute lookup.
Avoids importing heavy submodules at package import time and helps prevent circular imports (e.g. when `utils` imports `ops.operations_gguf`).

Symbols (top-level; keep in sync; no ghosts):
- `_SUBMODULES` (constant): Ordered list of candidate module paths searched for requested attributes.
- `_CACHE` (constant): Cache mapping attribute name -> resolved object.
- `__getattr__` (function): Import-time indirection that searches `_SUBMODULES` for the requested name.
- `__all__` (constant): Explicit export list (intentionally empty; exports are discovered dynamically).
"""

_SUBMODULES = (
    "apps.backend.runtime.ops.operations",
    "apps.backend.runtime.ops.operations_gguf",
)

_CACHE = {}


def __getattr__(name: str):  # pragma: no cover - import-time indirection
    if name in _CACHE:
        return _CACHE[name]
    import importlib
    for modpath in _SUBMODULES:
        mod = importlib.import_module(modpath)
        if hasattr(mod, name):
            obj = getattr(mod, name)
            _CACHE[name] = obj
            return obj
    raise AttributeError(name)


__all__ = []
