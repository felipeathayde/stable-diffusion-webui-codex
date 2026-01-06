"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Legacy-facing CLIP wrapper backed by the Codex patcher stack.
Provides a small `CLIP` facade exposing `cond_stage_model`, `tokenizer`, and a `ModelPatcher` with device/offload defaults.

Symbols (top-level; keep in sync; no ghosts):
- `JointTextEncoder` (class): Thin `ModuleDict` wrapper used to hold text encoder weights as a joint stage model.
- `CLIP` (class): Wrapper around text encoder + tokenizer with patching helpers and cloning support.
"""

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from .base import ModelPatcher
from apps.backend.runtime.nn import ModuleDict, ObjectDict


class JointTextEncoder(ModuleDict):
    pass


class CLIP:
    def __init__(self, model_dict=None, tokenizer_dict=None, *, model_config=None, no_init=False):
        model_dict = model_dict or {}
        tokenizer_dict = tokenizer_dict or {}
        if no_init:
            return

        load_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
        offload_device = memory_management.manager.get_offload_device(DeviceRole.TEXT_ENCODER)

        self.cond_stage_model = JointTextEncoder(model_dict)
        if model_config is not None:
            setattr(self.cond_stage_model, "model_config", model_config)
        self.tokenizer = ObjectDict(tokenizer_dict)
        self.patcher = ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        return n

    def add_patches(self, *arg, **kwargs):
        return self.patcher.add_patches(*arg, **kwargs)
