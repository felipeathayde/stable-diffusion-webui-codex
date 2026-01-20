"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Minimal registry for ControlNet module constructors.
Provides a default registry instance used by patchers to resolve and construct ControlNet module implementations.

Symbols (top-level; keep in sync; no ghosts):
- `ControlArchitectureRegistry` (class): Registry mapping architecture identifiers to constructors.
- `default_architecture_registry` (constant): Default registry instance used by the ControlNet patcher stack.
- `resolve_control_module` (function): Resolves an architecture constructor by name.
- `create_control_module` (function): Instantiates an architecture by name.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional


class ControlArchitectureRegistry:
    """Registry mapping architecture identifiers to constructors."""

    def __init__(self) -> None:
        self._constructors: Dict[str, Callable[..., object]] = {}

    def register(self, name: str, constructor: Callable[..., object]) -> None:
        if name in self._constructors:
            raise ValueError(f"Control architecture '{name}' already registered")
        self._constructors[name] = constructor

    def resolve(self, name: str) -> Callable[..., object]:
        try:
            return self._constructors[name]
        except KeyError as exc:
            raise KeyError(f"Unknown control architecture '{name}'") from exc

    def available(self) -> Dict[str, Callable[..., object]]:
        return dict(self._constructors)


default_architecture_registry = ControlArchitectureRegistry()


def resolve_control_module(name: str, registry: Optional[ControlArchitectureRegistry] = None) -> Callable[..., object]:
    registry = registry or default_architecture_registry
    return registry.resolve(name)


def create_control_module(name: str, *args, registry: Optional[ControlArchitectureRegistry] = None, **kwargs) -> object:
    constructor = resolve_control_module(name, registry=registry)
    return constructor(*args, **kwargs)
