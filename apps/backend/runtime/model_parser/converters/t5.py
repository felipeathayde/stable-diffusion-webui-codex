"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: T5/UMT5 state-dict conversion helpers for model parsing.
Applies lightweight normalization rules (currently: enforce float32 layer-norm weights for numerical stability) for T5-family text encoders.

Symbols (top-level; keep in sync; no ghosts):
- `_clone` (function): Makes a shallow copy of a state dict mapping.
- `_ensure_layer_norm_dtype` (function): Ensures a layer-norm weight tensor is float32.
- `convert_t5_encoder` (function): Normalizes a T5 encoder state dict (layer norm dtype).
- `convert_t5xxl_encoder` (function): Alias for `convert_t5_encoder` (T5-XXL).
- `convert_umt5_encoder` (function): Alias for `convert_t5_encoder` (UMT5).
"""

from __future__ import annotations

from typing import Any, Dict

import torch


def _clone(sd: Dict[str, Any]) -> Dict[str, Any]:
    return dict(sd)


def _ensure_layer_norm_dtype(sd: Dict[str, Any], key: str) -> None:
    tensor = sd.get(key)
    if isinstance(tensor, torch.Tensor) and tensor.dtype != torch.float32:
        sd[key] = tensor.to(torch.float32)


def convert_t5_encoder(sd: Dict[str, Any]) -> Dict[str, Any]:
    work = _clone(sd)
    # Ensure final layer norm remains in float32 for numerical stability.
    _ensure_layer_norm_dtype(work, "transformer.encoder.final_layer_norm.weight")
    _ensure_layer_norm_dtype(work, "transformer.final_layer_norm.weight")
    return work


def convert_t5xxl_encoder(sd: Dict[str, Any]) -> Dict[str, Any]:
    return convert_t5_encoder(sd)


def convert_umt5_encoder(sd: Dict[str, Any]) -> Dict[str, Any]:
    return convert_t5_encoder(sd)


__all__ = ["convert_t5_encoder", "convert_t5xxl_encoder", "convert_umt5_encoder"]
