from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

import torch

from apps.backend.runtime.utils import ParameterGGUF
from apps.backend.runtime.model_registry.specs import QuantizationHint, QuantizationKind
from .errors import ValidationError


def detect_quantization_from_tensors(tensors: Iterable[object]) -> QuantizationHint:
    has_fp4 = False
    for value in tensors:
        if isinstance(value, ParameterGGUF):
            return QuantizationHint(kind=QuantizationKind.GGUF, detail="parameter_gguf")
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float8_e4m3fn:
                has_fp4 = True
            elif value.dtype == torch.float8_e5m2:
                has_fp4 = True
        if isinstance(value, Mapping):
            nested = detect_quantization_from_tensors(value.values())
            if nested.kind != QuantizationKind.NONE:
                return nested
    if has_fp4:
        return QuantizationHint(kind=QuantizationKind.FP4)
    return QuantizationHint()


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
            if isinstance(value, ParameterGGUF):
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
    "validate_component_dtypes",
]
