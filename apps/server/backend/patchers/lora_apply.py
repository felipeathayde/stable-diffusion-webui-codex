from __future__ import annotations

"""
Native LoRA application pipeline (no legacy modules).

This layer converts LoRA files into patch dictionaries and applies them to the
engine's UNet and text encoders via our ModelPatcher system.

It builds on the existing in-backend patchers (UnetPatcher/CLIP ModelPatcher)
and the LoRA weight merge helpers in patchers/lora.py (which support multiple
LoRA variants and quantization-aware merges).
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Iterable, List

import torch
import safetensors.torch as sf

from .lora import model_lora_keys_unet, model_lora_keys_clip, load_lora


@dataclass
class AppliedStats:
    files: int = 0
    params_touched: int = 0


def _build_to_load_maps(engine) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return LoRA-key → model-param maps for UNet and CLIP encoders."""
    unet_model = engine.forge_objects_after_applying_lora.unet.model
    clip_model = engine.forge_objects_after_applying_lora.clip.cond_stage_model
    _sdk_u, unet_map = model_lora_keys_unet(unet_model)
    _sdk_c, clip_map = model_lora_keys_clip(clip_model)
    return unet_map, clip_map


def _apply_patches(patcher, filename: str, patch_dict: Dict[str, Any], strength: float) -> int:
    """Add patches to a ModelPatcher and return number of matched parameters."""
    # The patchers expect a flattened dict of model_key -> patch tuple(s)
    touched = 0
    if not patch_dict:
        return 0
    matched = patcher.add_patches(filename=filename, patches=patch_dict, strength_patch=float(strength), strength_model=1.0)
    touched += len(matched)
    return touched


def apply_loras_to_engine(engine, selections: Iterable[dict | Any]) -> AppliedStats:
    """Apply a list of LoRA selections to the engine (UNet + CLIP).

    Each selection item must carry `path` and optional `weight` fields.
    """
    stats = AppliedStats()
    if not selections:
        return stats

    unet_map, clip_map = _build_to_load_maps(engine)
    unet_patcher = engine.forge_objects_after_applying_lora.unet
    clip_patcher = engine.forge_objects_after_applying_lora.clip.patcher

    for sel in selections:
        path = str(getattr(sel, "path", None) or sel.get("path"))  # type: ignore[attr-defined]
        if not path:
            continue
        weight = float(getattr(sel, "weight", None) if hasattr(sel, "weight") else sel.get("weight", 1.0))  # type: ignore[attr-defined]

        # Load weights once
        try:
            tensor_map = sf.load_file(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA '{path}': {e}")

        # Build per-model patch dictionaries
        unet_patch, _ = load_lora(tensor_map, to_load=unet_map)
        clip_patch, _ = load_lora(tensor_map, to_load=clip_map)

        # Apply to patchers (record how many keys matched)
        stats.params_touched += _apply_patches(unet_patcher, filename=path, patch_dict=unet_patch, strength=weight)
        stats.params_touched += _apply_patches(clip_patcher, filename=path, patch_dict=clip_patch, strength=weight)
        stats.files += 1

    # Materialize merges onto actual model parameters
    unet_patcher.refresh_loras()
    clip_patcher.refresh_loras()

    return stats


__all__ = ["apply_loras_to_engine", "AppliedStats"]

