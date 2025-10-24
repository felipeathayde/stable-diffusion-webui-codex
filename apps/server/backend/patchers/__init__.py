"""Patchers for backend models (CLIP, VAE, UNet, ControlNet, LoRA)."""

from .base import (
    ModelPatcher,
    set_model_options_patch_replace,
    set_model_options_post_cfg_function,
    set_model_options_pre_cfg_function,
)
from .clip import CLIP
from .clipvision import clip_preprocess, CLIP_VISION_G, CLIP_VISION_H, CLIP_VISION_VITL, Output
from .controlnet import ControlNet, ControlLora, T2IAdapter, apply_controlnet_advanced, load_t2i_adapter
from .lora import (
    LoraLoader,
    extra_weight_calculators,
    lora_collection_priority,
    model_lora_keys_clip,
    model_lora_keys_unet,
    load_lora,
    merge_lora_to_weight,
)
from .unet import UnetPatcher
from .vae import VAE

__all__ = [name for name in globals() if not name.startswith("_")]
