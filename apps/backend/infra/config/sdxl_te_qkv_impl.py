"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SDXL text-encoder QKV implementation selection (auto|split|fused).
Centralizes parsing of `CODEX_SDXL_TE_QKV_IMPL` into a typed enum so SDXL loader/keymap choices remain strict and consistent.

Symbols (top-level; keep in sync; no ghosts):
- `ENV_SDXL_TE_QKV_IMPL` (constant): Environment variable name controlling SDXL TE QKV layout selection.
- `SdxlTeQkvImpl` (enum): Supported QKV implementations (`auto`, `split`, `fused`).
- `DEFAULT_SDXL_TE_QKV_IMPL` (constant): Default QKV impl when unset (auto).
- `parse_sdxl_te_qkv_impl` (function): Parses a string into `SdxlTeQkvImpl` (strict; raises on invalid).
- `read_sdxl_te_qkv_impl` (function): Reads QKV impl from an env mapping (strict; raises on invalid).
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Mapping, Optional

from .bootstrap_env import get_bootstrap_env

ENV_SDXL_TE_QKV_IMPL = "CODEX_SDXL_TE_QKV_IMPL"
_REMOVED_ENV_SDXL_TE_FUSED_QKV = "CODEX_SDXL_TE_FUSED_QKV"


class SdxlTeQkvImpl(Enum):
    """How SDXL CLIP text encoders represent Q/K/V projections."""

    AUTO = "auto"   # keep native layout per TE (OpenCLIP=fused; HF/Codex=split)
    SPLIT = "split" # force split Q/K/V params (q_proj/k_proj/v_proj)
    FUSED = "fused" # force fused in_proj params (in_proj)


DEFAULT_SDXL_TE_QKV_IMPL = SdxlTeQkvImpl.AUTO


def parse_sdxl_te_qkv_impl(raw: str) -> SdxlTeQkvImpl:
    value = str(raw).strip().lower()
    for mode in SdxlTeQkvImpl:
        if mode.value == value:
            return mode
    allowed = ", ".join(m.value for m in SdxlTeQkvImpl)
    raise ValueError(f"{ENV_SDXL_TE_QKV_IMPL} must be one of: {allowed}; got: {raw!r}")


def read_sdxl_te_qkv_impl(env: Optional[Mapping[str, str]] = None) -> SdxlTeQkvImpl:
    """Return the configured SDXL TE QKV implementation.

    Precedence:
    - bootstrap override (resolved CLI) when env is None
    - env[CODEX_SDXL_TE_QKV_IMPL] if set (strict)
    - DEFAULT_SDXL_TE_QKV_IMPL
    """

    # Fail loud when a removed env var is still present to avoid silent confusion.
    removed_raw = get_bootstrap_env(_REMOVED_ENV_SDXL_TE_FUSED_QKV)
    if removed_raw is None and env is not None:
        removed_raw = env.get(_REMOVED_ENV_SDXL_TE_FUSED_QKV)
    if removed_raw is None and env is None:
        removed_raw = os.getenv(_REMOVED_ENV_SDXL_TE_FUSED_QKV)
    if removed_raw is not None and str(removed_raw).strip():
        raise ValueError(
            f"{_REMOVED_ENV_SDXL_TE_FUSED_QKV} was removed; use {ENV_SDXL_TE_QKV_IMPL}=fused instead."
        )

    env_map = os.environ if env is None else env
    raw = None if env is not None else get_bootstrap_env(ENV_SDXL_TE_QKV_IMPL)
    if raw is None:
        raw = env_map.get(ENV_SDXL_TE_QKV_IMPL)
    if raw is None:
        return DEFAULT_SDXL_TE_QKV_IMPL
    text = str(raw).strip()
    if not text:
        return DEFAULT_SDXL_TE_QKV_IMPL
    return parse_sdxl_te_qkv_impl(text)


__all__ = [
    "DEFAULT_SDXL_TE_QKV_IMPL",
    "ENV_SDXL_TE_QKV_IMPL",
    "SdxlTeQkvImpl",
    "parse_sdxl_te_qkv_impl",
    "read_sdxl_te_qkv_impl",
]
