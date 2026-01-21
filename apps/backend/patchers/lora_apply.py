"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native LoRA application pipeline (no legacy modules).
Converts LoRA files into patch dictionaries and applies them to the engine's UNet and CLIP via the `ModelPatcher` system, then materializes
LoRA application by refreshing LoRAs on the patchers (merge default; optional on-the-fly via `CODEX_LORA_APPLY_MODE=online`).

Symbols (top-level; keep in sync; no ghosts):
- `AppliedStats` (dataclass): Counters for applied LoRA files and matched parameters.
- `_build_to_load_maps` (function): Builds LoRA-key → model-param maps for UNet and CLIP encoders.
- `_apply_patches` (function): Adds patches to a patcher and returns the number of matched parameters.
- `apply_loras_to_engine` (function): Applies selected LoRAs to the engine's patchers and refreshes LoRA application (merge or online).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Iterable

import safetensors.torch as sf

from apps.backend.infra.config.lora_apply_mode import LoraApplyMode, read_lora_apply_mode

from .lora import model_lora_keys_unet, model_lora_keys_clip, load_lora


@dataclass
class AppliedStats:
    files: int = 0
    params_touched: int = 0


def _build_to_load_maps(engine) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return LoRA-key → model-param maps for UNet and CLIP encoders."""
    unet_model = engine.codex_objects_after_applying_lora.denoiser.model
    clip_model = engine.codex_objects_after_applying_lora.text_encoders["clip"].cond_stage_model
    unet_map = model_lora_keys_unet(unet_model)
    clip_map = model_lora_keys_clip(clip_model)
    return unet_map, clip_map


def _apply_patches(patcher, filename: str, patch_dict: Dict[str, Any], strength: float, *, online_mode: bool) -> int:
    """Add patches to a ModelPatcher and return number of matched parameters."""
    # The patchers expect a flattened dict of model_key -> patch tuple(s)
    touched = 0
    if not patch_dict:
        return 0
    matched = patcher.add_patches(
        filename=filename,
        patches=patch_dict,
        strength_patch=float(strength),
        strength_model=1.0,
        online_mode=online_mode,
    )
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
    unet_patcher = engine.codex_objects_after_applying_lora.denoiser
    clip_patcher = engine.codex_objects_after_applying_lora.text_encoders["clip"].patcher

    apply_mode = read_lora_apply_mode()
    online_mode = apply_mode == LoraApplyMode.ONLINE

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
        stats.params_touched += _apply_patches(
            unet_patcher,
            filename=path,
            patch_dict=unet_patch,
            strength=weight,
            online_mode=online_mode,
        )
        stats.params_touched += _apply_patches(
            clip_patcher,
            filename=path,
            patch_dict=clip_patch,
            strength=weight,
            online_mode=online_mode,
        )
        stats.files += 1

    # Materialize merges onto actual model parameters
    unet_patcher.refresh_loras()
    clip_patcher.refresh_loras()

    return stats


__all__ = ["apply_loras_to_engine", "AppliedStats"]
