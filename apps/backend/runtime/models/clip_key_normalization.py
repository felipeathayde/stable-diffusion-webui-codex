"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: CLIP state-dict key normalization facade for runtime loaders.
Delegates CLIP key-style detection/remapping to canonical keymap ownership (`runtime/state_dict/keymap_sdxl_clip.py`) and returns
materialized normalized mappings plus resolved layout metadata used by layout-aware module selection.

Symbols (top-level; keep in sync; no ghosts):
- `normalize_codex_clip_state_dict_with_layout` (function): Normalizes a CLIP state dict and returns resolved layout metadata.
- `normalize_codex_clip_state_dict` (function): Backward-compatible convenience wrapper returning only normalized tensors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Literal

from apps.backend.runtime.state_dict.keymap_sdxl_clip import (
    ClipLayoutMetadata,
    remap_clip_state_dict_with_layout,
)

_QKVImpl = Literal["auto", "split", "fused"]
_ProjectionOrientationTarget = Literal["auto", "linear", "matmul"]


def normalize_codex_clip_state_dict_with_layout(
    state_dict: Mapping[str, Any],
    *,
    num_layers: int,
    keep_projection: bool,
    qkv_impl: _QKVImpl = "auto",
    projection_orientation: _ProjectionOrientationTarget = "auto",
    layout_metadata: ClipLayoutMetadata | None = None,
    require_projection: bool = False,
) -> tuple[Dict[str, Any], ClipLayoutMetadata]:
    _style, resolved_layout, remapped = remap_clip_state_dict_with_layout(
        dict(state_dict),
        num_layers=num_layers,
        keep_projection=keep_projection,
        qkv_impl=qkv_impl,
        projection_orientation=projection_orientation,
        layout_metadata=layout_metadata,
        require_projection=require_projection,
    )
    return dict(remapped.items()), resolved_layout


def normalize_codex_clip_state_dict(
    state_dict: Mapping[str, Any],
    *,
    num_layers: int,
    keep_projection: bool,
    qkv_impl: _QKVImpl = "auto",
    projection_orientation: _ProjectionOrientationTarget = "auto",
    layout_metadata: ClipLayoutMetadata | None = None,
    require_projection: bool = False,
) -> Dict[str, Any]:
    normalized, _layout = normalize_codex_clip_state_dict_with_layout(
        state_dict,
        num_layers=num_layers,
        keep_projection=keep_projection,
        qkv_impl=qkv_impl,
        projection_orientation=projection_orientation,
        layout_metadata=layout_metadata,
        require_projection=require_projection,
    )
    return normalized
