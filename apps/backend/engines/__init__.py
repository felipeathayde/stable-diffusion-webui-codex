"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Engine package facade and default registration entry points.
Exposes `register_default_engines(...)` and lazily resolves optional/large engine classes to avoid heavy imports during startup.

Symbols (top-level; keep in sync; no ghosts):
- `EngineLoadError` (class): Error raised when an engine fails to load required resources.
- `EngineExecutionError` (class): Error raised when an engine fails during inference execution.
- `register_default_engines` (function): Registers the canonical engine set into the registry.
- `_ENGINE_EXPORTS` (constant): Lazy export map `{name: (module_path, attr)}` for optional engines.
- `__getattr__` (function): Lazy import hook for engine class exports.
- `__all__` (constant): Explicit export list for the engine facade.
"""

from __future__ import annotations

# tags: backend, engines, lazy-imports

from importlib import import_module

from apps.backend.core.exceptions import EngineExecutionError, EngineLoadError
from apps.backend.core.registry import EngineRegistry


def register_default_engines(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    """Register the canonical set of engines into the provided registry."""

    registration = import_module("apps.backend.engines.registration")

    registration.register_sd15(registry=registry, replace=replace)
    registration.register_sdxl(registry=registry, replace=replace)
    registration.register_flux(registry=registry, replace=replace)
    registration.register_kontext(registry=registry, replace=replace)
    registration.register_sd20(registry=registry, replace=replace)
    registration.register_sd35(registry=registry, replace=replace)
    registration.register_zimage(registry=registry, replace=replace)
    # Optional engines are not auto-registered in strict mode (no silent fallbacks)
    registration.register_wan22_videos(registry=registry, replace=replace)


__all__ = [
    "EngineLoadError",
    "EngineExecutionError",
    "register_default_engines",
    "Wan2214BEngine",
    "Wan225BEngine",
]

_ENGINE_EXPORTS = {
    "Wan2214BEngine": ("apps.backend.engines.wan22.wan22_14b", "Wan2214BEngine"),
    "Wan225BEngine": ("apps.backend.engines.wan22.wan22_5b", "Wan225BEngine"),
}

def __getattr__(name: str):  # pragma: no cover - runtime dispatch
    if name in _ENGINE_EXPORTS:
        module_name, attr = _ENGINE_EXPORTS[name]
        mod = import_module(module_name)
        value = getattr(mod, attr)
        globals()[name] = value
        return value

    raise AttributeError(name)
