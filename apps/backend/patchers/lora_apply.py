"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native LoRA application pipeline (no legacy modules).
Converts LoRA files into patch dictionaries and applies them to the engine's denoiser and text encoders via the `ModelPatcher` system, then materializes LoRA application by refreshing LoRAs on the patchers (merge default; optional on-the-fly via `CODEX_LORA_APPLY_MODE=online`).
Fails loud when selected LoRAs do not match any runtime parameters (unsupported/incompatible key layout).
Patch dictionary keys may be plain parameter names or `(parameter, offset)` tuples for slice patches (e.g. fused-QKV text encoders).

Symbols (top-level; keep in sync; no ghosts):
- `AppliedStats` (dataclass): Counters for applied LoRA files and matched parameters.
- `_unwrap_patcher` (function): Returns a `ModelPatcher` from canonical text-encoder handles (`.patcher` required).
- `_collect_text_encoder_patchers` (function): Collects resettable text-encoder patchers keyed by encoder name.
- `_clear_and_refresh_lora_state` (function): Clears `lora_patches` and refreshes a patcher with fail-loud contract checks.
- `_refresh_lora_state` (function): Refreshes LoRA-merged weights on a patcher with fail-loud contract checks.
- `_build_to_load_maps` (function): Builds LoRA-key → model patch-target maps for UNet and CLIP encoders.
- `_apply_patches` (function): Adds patches to a patcher and returns the number of matched parameters.
- `apply_loras_to_engine` (function): Applies selected LoRAs to the engine's patchers and refreshes LoRA application (merge or online).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Iterable

import safetensors.torch as sf

from apps.backend.infra.config.lora_apply_mode import LoraApplyMode, read_lora_apply_mode
from apps.backend.runtime.adapters.base import PatchTarget

from .lora import model_lora_keys_unet, model_lora_keys_clip, load_lora


@dataclass
class AppliedStats:
    files: int = 0
    params_touched: int = 0


def _unwrap_patcher(entry: Any, *, label: str) -> Any:
    """Return a patcher from canonical text-encoder handles."""

    try:
        patcher = entry.patcher
    except AttributeError as exc:
        raise RuntimeError(
            "LoRA application requires canonical TextEncoderHandle entries with `.patcher` "
            f"(missing for {label})."
        ) from exc
    if patcher is None:
        raise RuntimeError(
            "LoRA application requires canonical TextEncoderHandle entries with non-null patcher "
            f"(missing for {label})."
        )
    if not hasattr(patcher, "add_patches") or not hasattr(patcher, "refresh_loras"):
        raise RuntimeError(
            f"LoRA application requires a patcher with add_patches/refresh_loras for {label}."
        )
    return patcher


def _collect_text_encoder_patchers(text_encoders: Any) -> Dict[str, Any]:
    """Collect text-encoder patchers keyed by encoder id."""

    if not isinstance(text_encoders, Mapping):
        return {}
    patchers: Dict[str, Any] = {}
    for key, entry in text_encoders.items():
        patchers[str(key)] = _unwrap_patcher(entry, label=f"text_encoders[{key!r}]")
    return patchers


def _clear_and_refresh_lora_state(patcher: Any, *, label: str) -> None:
    """Clear in-memory LoRA patch state and re-materialize merged weights."""

    if not hasattr(patcher, "lora_patches") or not hasattr(patcher, "refresh_loras"):
        raise RuntimeError(f"Engine exposes non-resettable LoRA patcher for {label}.")
    patcher.lora_patches = {}
    patcher.refresh_loras()


def _refresh_lora_state(patcher: Any, *, label: str) -> None:
    """Refresh merged LoRA weights without clearing patch definitions."""

    if not hasattr(patcher, "refresh_loras"):
        raise RuntimeError(f"Engine exposes non-refreshable LoRA patcher for {label}.")
    patcher.refresh_loras()


def _build_to_load_maps(engine) -> Tuple[Dict[str, PatchTarget], Dict[str, PatchTarget]]:
    """Return LoRA-key → model-param maps for UNet and CLIP encoders."""
    unet_model = engine.codex_objects_after_applying_lora.denoiser.model
    text_encoders = engine.codex_objects_after_applying_lora.text_encoders
    clip_entry = text_encoders.get("clip") if isinstance(text_encoders, Mapping) else None
    if clip_entry is None:
        raise RuntimeError("Engine does not expose required text_encoders['clip'] entry for LoRA key mapping.")
    try:
        clip_runtime = clip_entry.runtime
    except AttributeError as exc:
        raise RuntimeError(
            "Engine exposes non-canonical text_encoders['clip']; expected TextEncoderHandle with `.runtime`."
        ) from exc
    clip_model = getattr(clip_runtime, "cond_stage_model", None) if clip_runtime is not None else None
    if clip_model is None:
        raise RuntimeError(
            "Engine does not expose CLIP runtime cond_stage_model required for LoRA key mapping "
            "(expected text_encoders['clip'].runtime.cond_stage_model)."
        )
    unet_map = model_lora_keys_unet(unet_model)
    clip_map = model_lora_keys_clip(clip_model)
    return unet_map, clip_map


def _apply_patches(patcher, filename: str, patch_dict: Dict[PatchTarget, Any], strength: float, *, online_mode: bool) -> int:
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
    selected = list(selections or [])

    codex_objects = getattr(engine, "codex_objects_after_applying_lora", None)
    if codex_objects is None:
        raise RuntimeError("Engine is missing codex_objects_after_applying_lora required for LoRA application.")

    unet_patcher = getattr(codex_objects, "denoiser", None)
    text_encoders = getattr(codex_objects, "text_encoders", None)
    text_patchers = _collect_text_encoder_patchers(text_encoders)
    clip_patcher = text_patchers.get("clip")

    if not selected:
        if unet_patcher is None and not text_patchers:
            return stats
        if unet_patcher is None or not text_patchers:
            raise RuntimeError(
                "Engine exposes partial LoRA patcher state for empty selection reset "
                "(expected denoiser and at least one text encoder patcher)."
            )
        _clear_and_refresh_lora_state(unet_patcher, label="denoiser")
        for encoder_name, patcher in text_patchers.items():
            _clear_and_refresh_lora_state(patcher, label=f"text_encoders[{encoder_name!r}]")
        return stats

    if unet_patcher is None or clip_patcher is None:
        raise RuntimeError(
            "LoRA selections were provided, but the active engine does not expose required LoRA patchers "
            "(expected denoiser + text_encoders['clip'].patcher)."
        )

    unet_map, clip_map = _build_to_load_maps(engine)

    apply_mode = read_lora_apply_mode()
    online_mode = apply_mode == LoraApplyMode.ONLINE

    for sel in selected:
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
        if not unet_patch and not clip_patch:
            raise RuntimeError(
                "LoRA key layout mismatch: no compatible layers were found for "
                f"'{path}' on the active model keymap."
            )

        # Apply to patchers (record how many keys matched)
        unet_touched = _apply_patches(
            unet_patcher,
            filename=path,
            patch_dict=unet_patch,
            strength=weight,
            online_mode=online_mode,
        )
        clip_touched = _apply_patches(
            clip_patcher,
            filename=path,
            patch_dict=clip_patch,
            strength=weight,
            online_mode=online_mode,
        )
        touched_total = unet_touched + clip_touched
        if touched_total <= 0:
            raise RuntimeError(
                "LoRA apply mismatch: zero parameters were touched for "
                f"'{path}'. Verify LoRA/base-model compatibility and key layout."
            )
        stats.params_touched += touched_total
        stats.files += 1

    # Materialize merges onto actual model parameters
    _refresh_lora_state(unet_patcher, label="denoiser")
    _refresh_lora_state(clip_patcher, label="text_encoders['clip']")

    return stats


__all__ = ["apply_loras_to_engine", "AppliedStats"]
