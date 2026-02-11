"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Global policy for VAE layout lane selection (`auto|ldm_native|diffusers_native`).
Centralizes parsing of `CODEX_VAE_LAYOUT_LANE` so loader and engine override seams apply the same fail-loud layout contract.

Symbols (top-level; keep in sync; no ghosts):
- `ENV_VAE_LAYOUT_LANE` (constant): Environment variable controlling VAE layout lane policy.
- `VaeLayoutLane` (enum): Supported VAE lane modes (`auto`, `ldm_native`, `diffusers_native`).
- `DEFAULT_VAE_LAYOUT_LANE` (constant): Default mode when unset (`auto`).
- `parse_vae_layout_lane` (function): Parses a string into `VaeLayoutLane` (strict).
- `read_vae_layout_lane` (function): Reads lane mode from env/bootstrap (strict).
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Mapping, Optional

from .bootstrap_env import get_bootstrap_env

ENV_VAE_LAYOUT_LANE = "CODEX_VAE_LAYOUT_LANE"


class VaeLayoutLane(Enum):
    AUTO = "auto"
    LDM_NATIVE = "ldm_native"
    DIFFUSERS_NATIVE = "diffusers_native"


DEFAULT_VAE_LAYOUT_LANE = VaeLayoutLane.AUTO


def parse_vae_layout_lane(raw: str) -> VaeLayoutLane:
    value = str(raw).strip().lower()
    for mode in VaeLayoutLane:
        if mode.value == value:
            return mode
    allowed = ", ".join(mode.value for mode in VaeLayoutLane)
    raise ValueError(f"{ENV_VAE_LAYOUT_LANE} must be one of: {allowed}; got: {raw!r}")


def read_vae_layout_lane(env: Optional[Mapping[str, str]] = None) -> VaeLayoutLane:
    env_map = os.environ if env is None else env
    raw = None if env is not None else get_bootstrap_env(ENV_VAE_LAYOUT_LANE)
    if raw is None:
        raw = env_map.get(ENV_VAE_LAYOUT_LANE)
    if raw is None:
        return DEFAULT_VAE_LAYOUT_LANE
    text = str(raw).strip()
    if not text:
        return DEFAULT_VAE_LAYOUT_LANE
    return parse_vae_layout_lane(text)


__all__ = [
    "DEFAULT_VAE_LAYOUT_LANE",
    "ENV_VAE_LAYOUT_LANE",
    "VaeLayoutLane",
    "parse_vae_layout_lane",
    "read_vae_layout_lane",
]
