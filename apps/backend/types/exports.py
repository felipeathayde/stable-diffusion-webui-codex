"""Lazy export group definitions for backend __init__.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class LazyExports:
    """Groups of exports loaded lazily to avoid heavy import costs."""
    
    ENGINES: FrozenSet[str] = frozenset({
        "register_default_engines",
        "Wan2214BEngine",
        "Wan225BEngine",
        "WanI2V14BEngine",
        "WanT2V14BEngine",
    })
    
    TEXT_PROCESSING: FrozenSet[str] = frozenset({
        "ClassicTextProcessingEngine",
        "EmbeddingDatabase",
        "T5TextProcessingEngine",
        "embedding_from_b64",
        "embedding_to_b64",
        "text_emphasis",
        "text_parsing",
        "textual_inversion",
    })
    
    RUNTIME: FrozenSet[str] = frozenset({
        "attention",
        "logging",
        "memory_management",
        "models",
        "nn",
        "ops",
        "shared",
        "stream",
        "text_processing",
        "utils",
    })
    
    PATCHERS: FrozenSet[str] = frozenset({
        "CLIP",
        "ControlLora",
        "ControlNet",
        "LoraLoader",
        "ModelPatcher",
        "T2IAdapter",
        "UnetPatcher",
        "VAE",
        "apply_controlnet_advanced",
        "clip_preprocess",
        "extra_weight_calculators",
        "load_lora",
        "load_t2i_adapter",
        "merge_lora_to_weight",
        "model_lora_keys_clip",
        "model_lora_keys_unet",
        "set_model_options_patch_replace",
        "set_model_options_post_cfg_function",
        "set_model_options_pre_cfg_function",
    })
    
    SERVICES: FrozenSet[str] = frozenset({
        "ImageService",
        "MediaService",
        "OptionsService",
        "ProgressService",
        "SamplerService",
    })


# Singleton instance
LAZY_EXPORTS = LazyExports()

__all__ = ["LazyExports", "LAZY_EXPORTS"]
