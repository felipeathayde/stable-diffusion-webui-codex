"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Lightweight wrappers for dict-like module/config objects used by runtime stacks.

Symbols (top-level; keep in sync; no ghosts):
- `ModuleDict` (class): `torch.nn.Module` wrapper that registers a dict of submodules.
- `ObjectDict` (class): Simple attribute proxy exposing a dict via `obj.key` access.
- `Dummy` (class): Minimal `nn.Module` + `ConfigMixin` placeholder with a registered config.
"""

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch import nn


class ModuleDict(torch.nn.Module):
    def __init__(self, module_dict):
        super(ModuleDict, self).__init__()
        for name, module in module_dict.items():
            self.add_module(name, module)


class ObjectDict:
    def __init__(self, module_dict):
        for name, module in module_dict.items():
            setattr(self, name, module)


class Dummy(nn.Module, ConfigMixin):
    config_name = 'config.json'

    @register_to_config
    def __init__(self):
        super().__init__()
