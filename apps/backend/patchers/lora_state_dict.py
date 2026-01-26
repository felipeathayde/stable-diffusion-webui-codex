"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA tensor loading and target-key mapping helpers.
Wraps the Codex runtime LoRA adapter to parse tensors and build patch dictionaries for UNet/CLIP targets.
Patch targets may be plain parameter names or `(parameter, offset)` tuples for slice patches.

Symbols (top-level; keep in sync; no ghosts):
- `load_lora` (function): Loads/filters LoRA tensors for a target model mapping and returns normalized segments + tensor table.
- `model_lora_keys_clip` (function): Builds CLIP LoRA key mapping for a model.
- `model_lora_keys_unet` (function): Builds UNet LoRA key mapping for a model.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch

from apps.backend.runtime.adapters.base import PatchTarget
from apps.backend.runtime.adapters.lora import (
    convert_specs_to_patch_dict,
    model_lora_keys_clip as mapping_clip,
    model_lora_keys_unet as mapping_unet,
)
from apps.backend.runtime.adapters.lora.loader import parse_lora_tensors


def load_lora(
    lora_tensors: Mapping[str, torch.Tensor],
    to_load: Dict[str, PatchTarget],
) -> Tuple[Dict[PatchTarget, tuple], Dict[str, torch.Tensor]]:
    specs, loaded = parse_lora_tensors(lora_tensors, to_load)
    patch_dict = convert_specs_to_patch_dict(specs)
    remaining = {k: v for k, v in lora_tensors.items() if k not in loaded}
    return patch_dict, remaining


def model_lora_keys_clip(model, key_map=None):
    return mapping_clip(model, {} if key_map is None else key_map)


def model_lora_keys_unet(model, key_map=None):
    return mapping_unet(model, {} if key_map is None else key_map)


__all__ = [
    "load_lora",
    "model_lora_keys_clip",
    "model_lora_keys_unet",
]
