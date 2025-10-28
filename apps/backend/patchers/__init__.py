"""Patchers for backend models (CLIP, VAE, UNet, ControlNet, LoRA).

This package exposes many symbols via lazy imports to avoid circular imports
during application startup. Accessing an attribute will import the minimal
submodule that defines it.
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
