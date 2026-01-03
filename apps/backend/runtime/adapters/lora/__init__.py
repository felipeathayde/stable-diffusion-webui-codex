"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public exports for the native LoRA adapter pipeline.
Re-exports key mapping helpers and patch-dict builders used by patchers/engines to apply LoRAs to UNet and text encoders.

Symbols (top-level; keep in sync; no ghosts):
- `model_lora_keys_clip` (function): Builds LoRA key → CLIP/text-encoder parameter mapping.
- `model_lora_keys_unet` (function): Builds LoRA key → UNet parameter mapping (includes diffusers key normalization when available).
- `build_patch_dicts` (function): Builds patch dictionaries for a LoRA tensor mapping.
- `convert_specs_to_patch_dict` (function): Converts parsed `PatchSpec` entries into `ModelPatcher` patch dict tuples.
- `describe_lora_file` (function): Returns the set of LoRA variant labels detected in a LoRA file.
"""

from .mapping import model_lora_keys_clip, model_lora_keys_unet
from .pipeline import build_patch_dicts, describe_lora_file, convert_specs_to_patch_dict

__all__ = [
    "model_lora_keys_clip",
    "model_lora_keys_unet",
    "build_patch_dicts",
    "convert_specs_to_patch_dict",
    "describe_lora_file",
]
