"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Stable Diffusion ControlNet architecture implementations (and placeholders).
Exports the SD ControlNet module, LoRA-backed ControlNet, and T2I-Adapter integration used by the ControlNet patcher stack.

Symbols (top-level; keep in sync; no ghosts):
- `ControlNet` (class): Stable Diffusion ControlNet module implementation.
- `ControlNetLite` (class): Placeholder class for ControlNet-Lite variants (not yet ported).
- `ControlLiteConfig` (dataclass): Placeholder config for ControlNet-Lite variants.
- `ControlLora` (class): ControlNet LoRA module that materialises a ControlNet on demand.
- `T2IAdapter` (class): Adapter-based control module for T2I-Adapter weights.
- `load_t2i_adapter` (function): Loads a T2I-Adapter state dict into a runnable module.
- `__all__` (constant): Explicit export list for the SD architecture package.
"""

from .control import ControlNet
from .control_lite import ControlNetLite, ControlLiteConfig
from .lora import ControlLora
from .t2i_adapter import T2IAdapter, load_t2i_adapter

__all__ = [
    "ControlNet",
    "ControlNetLite",
    "ControlLiteConfig",
    "ControlLora",
    "T2IAdapter",
    "load_t2i_adapter",
]
