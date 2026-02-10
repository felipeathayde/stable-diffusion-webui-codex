"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Global policy for structural weight-conversion behavior (`auto|convert`).
Centralizes parsing of `CODEX_WEIGHT_STRUCTURAL_CONVERSION` so all keymap/converter seams can apply the same fail-loud rule.

Symbols (top-level; keep in sync; no ghosts):
- `ENV_WEIGHT_STRUCTURAL_CONVERSION` (constant): Environment variable controlling structural conversion policy.
- `WeightStructuralConversionMode` (enum): Supported modes (`auto`, `convert`).
- `DEFAULT_WEIGHT_STRUCTURAL_CONVERSION_MODE` (constant): Default mode when unset (`auto`).
- `parse_weight_structural_conversion_mode` (function): Parses a string into `WeightStructuralConversionMode` (strict).
- `read_weight_structural_conversion_mode` (function): Reads mode from env/bootstrap (strict).
- `is_structural_weight_conversion_enabled` (function): True only when policy is explicit `convert`.
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Mapping, Optional

from .bootstrap_env import get_bootstrap_env

ENV_WEIGHT_STRUCTURAL_CONVERSION = "CODEX_WEIGHT_STRUCTURAL_CONVERSION"


class WeightStructuralConversionMode(Enum):
    AUTO = "auto"
    CONVERT = "convert"


DEFAULT_WEIGHT_STRUCTURAL_CONVERSION_MODE = WeightStructuralConversionMode.AUTO


def parse_weight_structural_conversion_mode(raw: str) -> WeightStructuralConversionMode:
    value = str(raw).strip().lower()
    for mode in WeightStructuralConversionMode:
        if mode.value == value:
            return mode
    allowed = ", ".join(mode.value for mode in WeightStructuralConversionMode)
    raise ValueError(f"{ENV_WEIGHT_STRUCTURAL_CONVERSION} must be one of: {allowed}; got: {raw!r}")


def read_weight_structural_conversion_mode(
    env: Optional[Mapping[str, str]] = None,
) -> WeightStructuralConversionMode:
    env_map = os.environ if env is None else env
    raw = None if env is not None else get_bootstrap_env(ENV_WEIGHT_STRUCTURAL_CONVERSION)
    if raw is None:
        raw = env_map.get(ENV_WEIGHT_STRUCTURAL_CONVERSION)
    if raw is None:
        return DEFAULT_WEIGHT_STRUCTURAL_CONVERSION_MODE
    text = str(raw).strip()
    if not text:
        return DEFAULT_WEIGHT_STRUCTURAL_CONVERSION_MODE
    return parse_weight_structural_conversion_mode(text)


def is_structural_weight_conversion_enabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    return read_weight_structural_conversion_mode(env) is WeightStructuralConversionMode.CONVERT


__all__ = [
    "DEFAULT_WEIGHT_STRUCTURAL_CONVERSION_MODE",
    "ENV_WEIGHT_STRUCTURAL_CONVERSION",
    "WeightStructuralConversionMode",
    "is_structural_weight_conversion_enabled",
    "parse_weight_structural_conversion_mode",
    "read_weight_structural_conversion_mode",
]
