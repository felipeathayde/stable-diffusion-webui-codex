"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Small file/path helpers for ControlNet preprocessor weights.
Resolves the ControlNet cache root and loads weight state dicts via `safe_torch_load`.

Symbols (top-level; keep in sync; no ghosts):
- `controlnet_cache_root` (function): Returns the ControlNet weights cache root path (env override supported).
- `resolve_weights_file` (function): Resolves a relative weights path under the cache root or raises loudly.
- `load_state_dict` (function): Loads a state dict via `safe_torch_load` on CPU.
"""

from __future__ import annotations

import os
from pathlib import Path

from apps.backend.runtime.models.safety import safe_torch_load


def controlnet_cache_root() -> Path:
    root = os.environ.get("CODEX_CONTROLNET_CACHE")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".cache" / "codex" / "controlnet"


def resolve_weights_file(*relative: str) -> Path:
    root = controlnet_cache_root()
    path = root.joinpath(*relative)
    if not path.is_file():
        raise FileNotFoundError(
            f"ControlNet preprocessor weights not found: {path}.\n"
            "Populate the file manually or configure CODEX_CONTROLNET_CACHE."
        )
    return path


def load_state_dict(path: Path):
    return safe_torch_load(str(path), map_location="cpu")
