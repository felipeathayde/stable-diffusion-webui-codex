"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA refresh-signature mode selection for loader cache invalidation.
Controls whether refresh signatures use structural metadata only or include patch content hashing.

Symbols (top-level; keep in sync; no ghosts):
- `ENV_LORA_REFRESH_SIGNATURE` (constant): Environment variable controlling refresh signature mode.
- `LoraRefreshSignatureMode` (enum): Supported signature modes (`structural`, `content_sha256`).
- `DEFAULT_LORA_REFRESH_SIGNATURE_MODE` (constant): Default signature mode when unset.
- `parse_lora_refresh_signature_mode` (function): Parses a string into `LoraRefreshSignatureMode` (strict; raises on invalid).
- `read_lora_refresh_signature_mode` (function): Reads signature mode from env/bootstrap mapping (strict; raises on invalid).
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Mapping, Optional

from .bootstrap_env import get_bootstrap_env

ENV_LORA_REFRESH_SIGNATURE = "CODEX_LORA_REFRESH_SIGNATURE"


class LoraRefreshSignatureMode(Enum):
    """How LoRA loader refresh signatures are computed."""

    STRUCTURAL = "structural"
    CONTENT_SHA256 = "content_sha256"


DEFAULT_LORA_REFRESH_SIGNATURE_MODE = LoraRefreshSignatureMode.STRUCTURAL


def parse_lora_refresh_signature_mode(raw: str) -> LoraRefreshSignatureMode:
    value = str(raw).strip().lower()
    for mode in LoraRefreshSignatureMode:
        if mode.value == value:
            return mode
    allowed = ", ".join(mode.value for mode in LoraRefreshSignatureMode)
    raise ValueError(f"{ENV_LORA_REFRESH_SIGNATURE} must be one of: {allowed}; got: {raw!r}")


def read_lora_refresh_signature_mode(env: Optional[Mapping[str, str]] = None) -> LoraRefreshSignatureMode:
    env_map = os.environ if env is None else env
    raw = None if env is not None else get_bootstrap_env(ENV_LORA_REFRESH_SIGNATURE)
    if raw is None:
        raw = env_map.get(ENV_LORA_REFRESH_SIGNATURE)
    if raw is None:
        return DEFAULT_LORA_REFRESH_SIGNATURE_MODE
    text = str(raw).strip()
    if not text:
        return DEFAULT_LORA_REFRESH_SIGNATURE_MODE
    return parse_lora_refresh_signature_mode(text)


__all__ = [
    "DEFAULT_LORA_REFRESH_SIGNATURE_MODE",
    "ENV_LORA_REFRESH_SIGNATURE",
    "LoraRefreshSignatureMode",
    "parse_lora_refresh_signature_mode",
    "read_lora_refresh_signature_mode",
]
