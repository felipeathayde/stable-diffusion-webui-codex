"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Quantization detection and dtype validation for model-parser components.
Detects GGUF quantization via `CodexParameter` / CodexPack markers and detects NF4/FP4 key markers for fail-loud reporting, and provides a
strict validation helper to catch mis-detections where a component contains no floating-point tensors.

Symbols (top-level; keep in sync; no ghosts):
- `detect_quantization_from_tensors` (function): Recursively scans tensors/mappings to infer quantization kind (GGUF/none).
- `detect_state_dict_dtype` (function): Best-effort dtype / quantization hint for a state dict (returns a torch dtype or `"gguf"`).
- `detect_quantization_from_component` (function): Infers quantization from one component mapping (prefers GGUF tensor markers).
- `detect_quantization` (function): Infers quantization for a full parser context (UNet/transformer prioritized).
- `validate_component_dtypes` (function): Fails fast when a component has no floating-point tensors (likely a wrong split/prefix).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

import torch

from apps.backend.quantization.codexpack_tensor import CodexPackLinearQ4KTilepackV1Parameter
from apps.backend.quantization.tensor import CodexParameter
from apps.backend.runtime.model_registry.specs import QuantizationHint, QuantizationKind
from .errors import ValidationError


def detect_quantization_from_tensors(tensors: Iterable[object]) -> QuantizationHint:
    for value in tensors:
        if isinstance(value, CodexParameter) and value.qtype is not None:
            return QuantizationHint(kind=QuantizationKind.GGUF, detail="parameter_gguf")
        if isinstance(value, CodexPackLinearQ4KTilepackV1Parameter):
            return QuantizationHint(kind=QuantizationKind.GGUF, detail="parameter_codexpack")
        if isinstance(value, Mapping):
            nested = detect_quantization_from_tensors(value.values())
            if nested.kind != QuantizationKind.NONE:
                return nested
    return QuantizationHint()


def detect_state_dict_dtype(state_dict: Mapping[str, object]) -> torch.dtype | str:
    """Best-effort dtype / quantization hint for a state dict.

    - Returns ``"gguf"`` when the mapping contains CodexParameter packed weights.
    - Otherwise returns the first encountered torch dtype (defaults to fp32).
    """

    materialize = getattr(state_dict, "materialize", None)
    if callable(materialize):
        for value in state_dict.values():
            if isinstance(value, torch.Tensor):
                return value.dtype
        return torch.float32

    first_dtype: torch.dtype | None = None
    for idx, value in enumerate(state_dict.values()):
        if isinstance(value, CodexParameter) and value.qtype is not None:
            return "gguf"
        if isinstance(value, CodexPackLinearQ4KTilepackV1Parameter):
            return "gguf"
        if first_dtype is None and isinstance(value, torch.Tensor):
            first_dtype = value.dtype
        # Defensive cap: most GGUF dicts reveal themselves quickly.
        if idx >= 4096 and first_dtype is not None:
            break
    return first_dtype or torch.float32


def detect_quantization_from_component(component_state: Mapping[str, object]) -> QuantizationHint:
    has_nf4 = any("bitsandbytes__nf4" in key for key in component_state.keys())
    has_fp4 = any("bitsandbytes__fp4" in key for key in component_state.keys())

    hint = detect_quantization_from_tensors(component_state.values())
    if hint.kind != QuantizationKind.NONE:
        return hint
    if has_nf4:
        return QuantizationHint(kind=QuantizationKind.NF4, detail="key_marker")
    if has_fp4:
        return QuantizationHint(kind=QuantizationKind.FP4, detail="key_marker")
    return QuantizationHint()


def detect_quantization(context) -> QuantizationHint:
    # Prefer GGUF/NF4/FP4 detection from components (UNet first).
    priority_order = ["unet", "transformer", "text_encoder", "text_encoder_2", "text_encoder_3"]
    seen = set()
    for name in priority_order:
        component = context.components.get(name)
        if component is None:
            continue
        hint = detect_quantization_from_component(component.tensors)
        if hint.kind != QuantizationKind.NONE:
            return hint
        seen.add(name)

    # Fallback: scan all components that were not in priority order.
    for name, component in context.components.items():
        if name in seen:
            continue
        hint = detect_quantization_from_component(component.tensors)
        if hint.kind != QuantizationKind.NONE:
            return hint
    return QuantizationHint()


def validate_component_dtypes(context) -> None:
    for name, component in context.components.items():
        if not component.tensors:
            continue
        has_floating = False
        for value in component.tensors.values():
            if isinstance(value, CodexParameter) and value.qtype is not None:
                has_floating = True
                break
            if isinstance(value, CodexPackLinearQ4KTilepackV1Parameter):
                has_floating = True
                break
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                has_floating = True
                break
        if not has_floating:
            raise ValidationError(
                f"Component '{name}' has no floating-point tensors; possible mis-detection",
                component=name,
            )


__all__ = [
    "detect_quantization",
    "detect_quantization_from_component",
    "detect_state_dict_dtype",
    "validate_component_dtypes",
]
