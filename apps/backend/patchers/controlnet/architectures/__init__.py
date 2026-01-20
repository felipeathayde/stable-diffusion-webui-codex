"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: ControlNet architecture registry and built-in SD-family registrations.
Registers Codex-native ControlNet module implementations into a default registry at import time.

Symbols (top-level; keep in sync; no ghosts):
- `ControlArchitectureRegistry` (class): Registry mapping architecture identifiers to constructors.
- `default_architecture_registry` (constant): Default registry pre-populated with built-in architectures.
- `resolve_control_module` (function): Resolves an architecture constructor by name.
- `create_control_module` (function): Instantiates an architecture by name.
- `__all__` (constant): Explicit export list for the architecture facade.
"""

from .factory import (
    ControlArchitectureRegistry,
    default_architecture_registry,
    resolve_control_module,
    create_control_module,
)
from .sd.control import ControlNet
from .sd.control_lite import ControlNetLite
from .sd.lora import ControlLora
from .sd.t2i_adapter import T2IAdapter

# Register default SD-family modules.
default_architecture_registry.register("controlnet", ControlNet)
default_architecture_registry.register("controlnet_lite", ControlNetLite)
default_architecture_registry.register("controlnet_lora", ControlLora)
default_architecture_registry.register("t2i_adapter", T2IAdapter)

__all__ = [
    "ControlArchitectureRegistry",
    "default_architecture_registry",
    "resolve_control_module",
    "create_control_module",
]
