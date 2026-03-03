"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Online LoRA math selection (weight-merge vs activation-side).
Centralizes the meaning of the `--lora-online-math` flag used when `--lora-apply-mode=online`.

Symbols (top-level; keep in sync; no ghosts):
- `ENV_LORA_ONLINE_MATH` (constant): Environment variable controlling online LoRA math mode.
- `LoraOnlineMath` (enum): Supported online LoRA math modes (`weight_merge`, `activation`).
- `DEFAULT_LORA_ONLINE_MATH` (constant): Default online math mode when unset.
- `parse_lora_online_math` (function): Parses a string into `LoraOnlineMath` (strict; raises on invalid).
- `read_lora_online_math` (function): Reads online math mode from env/bootstrap mapping (strict; raises on invalid).
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Mapping, Optional

from .bootstrap_env import get_bootstrap_env

ENV_LORA_ONLINE_MATH = "CODEX_LORA_ONLINE_MATH"


class LoraOnlineMath(Enum):
    """How online LoRA patches are applied at runtime."""

    WEIGHT_MERGE = "weight_merge"
    ACTIVATION = "activation"


DEFAULT_LORA_ONLINE_MATH = LoraOnlineMath.WEIGHT_MERGE


def parse_lora_online_math(raw: str) -> LoraOnlineMath:
    value = str(raw).strip().lower()
    for mode in LoraOnlineMath:
        if mode.value == value:
            return mode
    allowed = ", ".join(m.value for m in LoraOnlineMath)
    raise ValueError(f"{ENV_LORA_ONLINE_MATH} must be one of: {allowed}; got: {raw!r}")


def read_lora_online_math(env: Optional[Mapping[str, str]] = None) -> LoraOnlineMath:
    env_map = os.environ if env is None else env
    raw = None if env is not None else get_bootstrap_env(ENV_LORA_ONLINE_MATH)
    if raw is None:
        raw = env_map.get(ENV_LORA_ONLINE_MATH)
    if raw is None:
        return DEFAULT_LORA_ONLINE_MATH
    text = str(raw).strip()
    if not text:
        return DEFAULT_LORA_ONLINE_MATH
    return parse_lora_online_math(text)


__all__ = [
    "DEFAULT_LORA_ONLINE_MATH",
    "ENV_LORA_ONLINE_MATH",
    "LoraOnlineMath",
    "parse_lora_online_math",
    "read_lora_online_math",
]
