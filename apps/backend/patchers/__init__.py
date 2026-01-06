"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Patcher package facade with lazy imports (CLIP/VAE/UNet/ControlNet/LoRA).
Exposes patcher symbols via `__getattr__` to avoid import cycles during startup; attributes resolve to the minimal submodule that defines them.

Symbols (top-level; keep in sync; no ghosts):
- `_EXPORTS` (constant): Export map `{symbol_name: (module_path, attribute_name)}` used by the lazy import hook.
- `__all__` (constant): Sorted list of public export names (keys of `_EXPORTS`).
- `__getattr__` (function): Lazy import hook that resolves exports on first access.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    # base
    "ModelPatcher": (".base", "ModelPatcher"),
    "set_model_options_patch_replace": (".base", "set_model_options_patch_replace"),
    "set_model_options_post_cfg_function": (".base", "set_model_options_post_cfg_function"),
    "set_model_options_pre_cfg_function": (".base", "set_model_options_pre_cfg_function"),
    # clip
    "CLIP": (".clip", "CLIP"),
    # clipvision
    "clip_preprocess": (".clipvision", "clip_preprocess"),
    "CLIP_VISION_G": (".clipvision", "CLIP_VISION_G"),
    "CLIP_VISION_H": (".clipvision", "CLIP_VISION_H"),
    "CLIP_VISION_VITL": (".clipvision", "CLIP_VISION_VITL"),
    "Output": (".clipvision", "Output"),
    # controlnet
    "ControlNet": (".controlnet", "ControlNet"),
    "ControlLora": (".controlnet", "ControlLora"),
    "T2IAdapter": (".controlnet", "T2IAdapter"),
    "apply_controlnet_advanced": (".controlnet", "apply_controlnet_advanced"),
    "load_t2i_adapter": (".controlnet", "load_t2i_adapter"),
    # lora
    "LoraLoader": (".lora", "LoraLoader"),
    "extra_weight_calculators": (".lora", "extra_weight_calculators"),
    "model_lora_keys_clip": (".lora", "model_lora_keys_clip"),
    "model_lora_keys_unet": (".lora", "model_lora_keys_unet"),
    "load_lora": (".lora", "load_lora"),
    "merge_lora_to_weight": (".lora", "merge_lora_to_weight"),
    "CodexLoraLoader": (".lora", "CodexLoraLoader"),
    # unet
    "UnetPatcher": (".unet", "UnetPatcher"),
    # vae
    "VAE": (".vae", "VAE"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:  # pragma: no cover - runtime behavior
    try:
        mod_path, sym = _EXPORTS[name]
    except KeyError as e:
        raise AttributeError(name) from e
    mod = import_module(mod_path, __name__)
    return getattr(mod, sym)
