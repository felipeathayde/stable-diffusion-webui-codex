"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA patch loader/applier for runtime models (UNet/CLIP and related components).
Loads LoRA tensors, builds patch dictionaries for target models, and provides merge/applier helpers used by the patcher system.

Symbols (top-level; keep in sync; no ghosts):
- `LoraPatchEntry` (type): Raw LoRA patch entry tuple/list shape used by conversion helpers.
- `LoraVariant` (enum): Supported LoRA patch variants (diff/set/lora/loha/lokr/glora) with tag parsing.
- `OffsetSpec` (dataclass): Tensor narrow/slice spec used for offset-based LoRA segments.
- `LoraPatchSegment` (dataclass): Normalized patch segment representation (variant + tensors + offsets) used by apply helpers.
- `extra_weight_calculators` (dict): Maps custom patch tags to calculator callables.
- `load_lora` (function): Loads/filters LoRA tensors for a target model mapping and returns normalized segments + tensor table.
- `model_lora_keys_clip` (function): Builds CLIP LoRA key mapping for a model.
- `model_lora_keys_unet` (function): Builds UNet LoRA key mapping for a model.
- `weight_decompose` (function): Decomposes weights to an apply-friendly form for certain variants/quant paths.
- `merge_lora_to_weight` (function): Merges a LoRA patch into a target weight tensor (variant-dispatched).
- `get_parameter_devices` (function): Captures current parameter device mapping for later restoration.
- `set_parameter_devices` (function): Restores parameters to a previously captured device mapping.
- `CodexLoraLoader` (class): High-level loader/applier that integrates mapping, device placement, and progress reporting (tqdm).
- `LoraLoader` (alias): Alias to `CodexLoraLoader` (legacy-facing).
"""

from __future__ import annotations

from .lora_loader import CodexLoraLoader, get_parameter_devices, set_parameter_devices
from .lora_merge import merge_lora_to_weight, weight_decompose
from .lora_registry import extra_weight_calculators
from .lora_state_dict import load_lora, model_lora_keys_clip, model_lora_keys_unet
from .lora_types import LoraPatchEntry, LoraPatchSegment, LoraVariant, OffsetSpec

LoraLoader = CodexLoraLoader

__all__ = [
    "CodexLoraLoader",
    "LoraLoader",
    "LoraPatchEntry",
    "LoraPatchSegment",
    "LoraVariant",
    "OffsetSpec",
    "extra_weight_calculators",
    "get_parameter_devices",
    "load_lora",
    "merge_lora_to_weight",
    "model_lora_keys_clip",
    "model_lora_keys_unet",
    "set_parameter_devices",
    "weight_decompose",
]

