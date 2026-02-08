"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Family-aware VAE normalization policy resolution for scale/shift semantics.
Defines typed family normalization parameters and resolves effective scalar normalization values from model config
plus family policy, including strict shift-factor contracts for families that do/do-not use shift.

Symbols (top-level; keep in sync; no ghosts):
- `VaeShiftMode` (enum): Shift contract for a family (`ABSENT` or `REQUIRED`).
- `VaeFamilyNormalizationParams` (dataclass): Normalization defaults/contract for a model family.
- `VaeNormalizationPolicy` (dataclass): Resolved normalization policy consumed by the VAE wrapper.
- `VAE_FAMILY_NORMALIZATION_PARAMS` (constant): Normalization params for every known runtime family.
- `read_vae_config_field` (function): Reads a config field with presence tracking across mapping/object configs.
- `get_vae_family_normalization_params` (function): Returns normalization params for a family (raises if missing).
- `resolve_vae_normalization_policy` (function): Resolves effective scale/shift policy with fail-loud validation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from apps.backend.runtime.model_registry.family_runtime import FAMILY_RUNTIME_SPECS
from apps.backend.runtime.model_registry.specs import ModelFamily


class VaeShiftMode(Enum):
    ABSENT = "absent"
    REQUIRED = "required"


@dataclass(frozen=True, slots=True)
class VaeFamilyNormalizationParams:
    family: ModelFamily
    scaling_factor: float
    shift_factor: float | None
    shift_mode: VaeShiftMode


@dataclass(frozen=True, slots=True)
class VaeNormalizationPolicy:
    scaling_factor: float
    shift_factor: float | None


_SHIFT_REQUIRED_FAMILIES = frozenset(
    {
        ModelFamily.SD3,
        ModelFamily.SD35,
        ModelFamily.FLUX,
        ModelFamily.FLUX_KONTEXT,
        ModelFamily.CHROMA,
        ModelFamily.ZIMAGE,
    }
)


def _build_family_normalization_params() -> dict[ModelFamily, VaeFamilyNormalizationParams]:
    params: dict[ModelFamily, VaeFamilyNormalizationParams] = {}
    for family, runtime_spec in FAMILY_RUNTIME_SPECS.items():
        if family in _SHIFT_REQUIRED_FAMILIES:
            shift_factor: float | None = float(runtime_spec.vae_shift_factor)
            shift_mode = VaeShiftMode.REQUIRED
        else:
            shift_factor = None
            shift_mode = VaeShiftMode.ABSENT
        params[family] = VaeFamilyNormalizationParams(
            family=family,
            scaling_factor=float(runtime_spec.vae_scaling_factor),
            shift_factor=shift_factor,
            shift_mode=shift_mode,
        )
    return params


VAE_FAMILY_NORMALIZATION_PARAMS = _build_family_normalization_params()


def get_vae_family_normalization_params(family: ModelFamily) -> VaeFamilyNormalizationParams:
    params = VAE_FAMILY_NORMALIZATION_PARAMS.get(family)
    if params is None:
        raise RuntimeError(f"No VAE normalization parameters registered for family '{family.value}'.")
    return params


def read_vae_config_field(config: object | None, key: str) -> tuple[bool, object | None]:
    if config is None:
        return False, None
    if isinstance(config, Mapping):
        if key in config:
            return True, config[key]
        return False, None
    if hasattr(config, key):
        try:
            return True, getattr(config, key)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read VAE config field '{key}' from attribute access.") from exc
    getter = getattr(config, "get", None)
    if callable(getter):
        sentinel = object()
        try:
            value = getter(key, sentinel)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read VAE config field '{key}' via mapping accessor.") from exc
        if value is sentinel:
            return False, None
        return True, value
    return False, None


def _coerce_finite_float(field: str, value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"VAE config field '{field}' must be a finite numeric value.") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"VAE config field '{field}' must be a finite numeric value.")
    return number


def _resolve_scaling_factor(
    *,
    config: object | None,
    family_params: VaeFamilyNormalizationParams | None,
) -> float:
    scale_present, scale_value = read_vae_config_field(config, "scaling_factor")
    if scale_present:
        if scale_value is None:
            raise RuntimeError(
                "VAE config field 'scaling_factor' is null; missing 'scaling_factor' is not allowed."
            )
        return _coerce_finite_float("scaling_factor", scale_value)
    if family_params is not None:
        return family_params.scaling_factor
    raise RuntimeError(
        "VAE config missing 'scaling_factor' and no family fallback available; engines require normalization semantics"
    )


def _resolve_shift_factor(
    *,
    config: object | None,
    family: ModelFamily | None,
    family_params: VaeFamilyNormalizationParams | None,
) -> float | None:
    shift_present, shift_value = read_vae_config_field(config, "shift_factor")
    if family_params is None:
        if not shift_present or shift_value is None:
            return None
        return _coerce_finite_float("shift_factor", shift_value)
    if family_params.shift_mode is VaeShiftMode.ABSENT:
        if shift_present and shift_value is not None:
            raise RuntimeError(
                f"VAE family '{family_params.family.value}' does not support shift_factor; "
                f"received {shift_value!r} in config."
            )
        return None
    if not shift_present or shift_value is None:
        family_name = family.value if family is not None else family_params.family.value
        raise RuntimeError(f"VAE family '{family_name}' requires shift_factor in config.")
    return _coerce_finite_float("shift_factor", shift_value)


def resolve_vae_normalization_policy(
    *,
    config: object | None,
    family: ModelFamily | None,
) -> VaeNormalizationPolicy:
    family_params = VAE_FAMILY_NORMALIZATION_PARAMS.get(family) if family is not None else None
    scaling_factor = _resolve_scaling_factor(config=config, family_params=family_params)
    shift_factor = _resolve_shift_factor(config=config, family=family, family_params=family_params)
    return VaeNormalizationPolicy(scaling_factor=scaling_factor, shift_factor=shift_factor)


__all__ = [
    "VaeShiftMode",
    "VaeFamilyNormalizationParams",
    "VaeNormalizationPolicy",
    "VAE_FAMILY_NORMALIZATION_PARAMS",
    "read_vae_config_field",
    "get_vae_family_normalization_params",
    "resolve_vae_normalization_policy",
]
