"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Lazy export name groups for backend package facades.
Defines frozen export sets used by backend `__getattr__`/`__all__` wiring to expose large subsystems on demand without heavy imports during startup.

Symbols (top-level; keep in sync; no ghosts):
- `LazyExports` (dataclass): Frozen sets of export names grouped by subsystem (engines/runtime/patchers/services/etc.).
- `LAZY_EXPORTS` (constant): Singleton instance of `LazyExports`.
- `__all__` (constant): Explicit export list for this module.
"""

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
