"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF stage selection and model loading.
Validates stage GGUF paths and loads stage weights into `WanTransformer2DModel` via Codex GGUF operations and WAN key remapping.

Symbols (top-level; keep in sync; no ghosts):
- `normalize_win_path` (function): Normalizes Windows drive paths to WSL-style `/mnt/<drive>/...` paths when running on non-Windows.
- `pick_stage_gguf` (function): Validates and returns the stage GGUF file path (strict: must be an explicit `.gguf` file).
- `load_stage_model_from_gguf` (function): Loads a stage GGUF into a runtime transformer (wraps ops + key remapping).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.checkpoint_io import _load_gguf_state_dict

from .diagnostics import get_logger
from .model import load_wan_transformer_from_state_dict, remap_wan22_gguf_state_dict


def normalize_win_path(path: str) -> str:
    if os.name == "nt":
        return path
    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        drive = path[0].lower()
        rest = path[2:].lstrip("\\/")
        return f"/mnt/{drive}/" + rest.replace("\\\\", "/").replace("\\", "/")
    return path


def pick_stage_gguf(dir_path: Optional[str], *, stage: str) -> Optional[str]:
    if not dir_path:
        return None

    raw = normalize_win_path(dir_path)
    abspath = raw if os.path.isabs(raw) else os.path.abspath(raw)
    if os.path.isfile(abspath) and abspath.lower().endswith(".gguf"):
        return abspath
    if os.path.isdir(abspath):
        raise RuntimeError(
            f"WAN22 GGUF stage '{stage}' requires an explicit .gguf file path (sha-selected); got directory: {abspath}"
        )
    return None


def load_stage_model_from_gguf(
    gguf_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    logger: Any,
):
    log = get_logger(logger)
    state = _load_gguf_state_dict(gguf_path)
    state = remap_wan22_gguf_state_dict(state)
    with using_codex_operations(device=device, dtype=dtype, bnb_dtype="gguf"):
        model = load_wan_transformer_from_state_dict(state, config=None)
    model.eval()
    log.info("[wan22.gguf] loaded stage model: %s", os.path.basename(gguf_path))
    return model
