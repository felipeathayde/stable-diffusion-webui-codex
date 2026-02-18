"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA merge-mode selection for offline weight materialization.
Centralizes parsing/reading for fast (`float32`) vs precise (`float64`) merge math.

Symbols (top-level; keep in sync; no ghosts):
- `ENV_LORA_MERGE_MODE` (constant): Environment variable controlling merge mode.
- `LoraMergeMode` (enum): Supported merge modes (`fast`, `precise`).
- `DEFAULT_LORA_MERGE_MODE` (constant): Default merge mode when unset.
- `parse_lora_merge_mode` (function): Parses a string into `LoraMergeMode` (strict; raises on invalid).
- `read_lora_merge_mode` (function): Reads merge mode from env/bootstrap mapping (strict; raises on invalid).
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Mapping, Optional

from .bootstrap_env import get_bootstrap_env

ENV_LORA_MERGE_MODE = "CODEX_LORA_MERGE_MODE"


class LoraMergeMode(Enum):
    """How offline LoRA merges are materialized."""

    FAST = "fast"
    PRECISE = "precise"


DEFAULT_LORA_MERGE_MODE = LoraMergeMode.FAST


def parse_lora_merge_mode(raw: str) -> LoraMergeMode:
    value = str(raw).strip().lower()
    for mode in LoraMergeMode:
        if mode.value == value:
            return mode
    allowed = ", ".join(mode.value for mode in LoraMergeMode)
    raise ValueError(f"{ENV_LORA_MERGE_MODE} must be one of: {allowed}; got: {raw!r}")


def read_lora_merge_mode(env: Optional[Mapping[str, str]] = None) -> LoraMergeMode:
    env_map = os.environ if env is None else env
    raw = None if env is not None else get_bootstrap_env(ENV_LORA_MERGE_MODE)
    if raw is None:
        raw = env_map.get(ENV_LORA_MERGE_MODE)
    if raw is None:
        return DEFAULT_LORA_MERGE_MODE
    text = str(raw).strip()
    if not text:
        return DEFAULT_LORA_MERGE_MODE
    return parse_lora_merge_mode(text)


__all__ = [
    "DEFAULT_LORA_MERGE_MODE",
    "ENV_LORA_MERGE_MODE",
    "LoraMergeMode",
    "parse_lora_merge_mode",
    "read_lora_merge_mode",
]
