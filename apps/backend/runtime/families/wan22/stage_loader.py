"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF stage selection and model loading.
Validates stage GGUF paths and loads stage weights into `WanTransformer2DModel` via Codex GGUF operations (`using_codex_operations(weight_format="gguf")`) and WAN key remapping, with device-targeted GGUF tensor loading.
Optionally applies a per-stage LoRA file (merge/online) for LightX2V-style stage patches.

Symbols (top-level; keep in sync; no ghosts):
- `pick_stage_gguf` (function): Validates and returns the stage GGUF file path (strict: must be an explicit `.gguf` file).
- `load_stage_model_from_gguf` (function): Loads a stage GGUF into a runtime transformer (device-aware GGUF load + key remapping + ops wrapper).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.checkpoint.io import load_gguf_state_dict

from .diagnostics import get_logger
from .model import load_wan_transformer_from_state_dict, remap_wan22_gguf_state_dict
from .paths import normalize_win_path
from .stage_lora import apply_wan22_stage_lora


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
    stage: str,
    device: torch.device,
    dtype: torch.dtype,
    lora_path: Optional[str] = None,
    lora_weight: Optional[float] = None,
    logger: Any,
):
    log = get_logger(logger)
    state = load_gguf_state_dict(gguf_path, device=device)
    state = remap_wan22_gguf_state_dict(state)
    with using_codex_operations(device=device, dtype=dtype, weight_format="gguf"):
        model = load_wan_transformer_from_state_dict(state, config=None)
    model.eval()
    apply_wan22_stage_lora(
        model,
        stage=stage,
        lora_path=lora_path,
        lora_weight=lora_weight,
        logger=logger,
    )
    log.info("[wan22.gguf] loaded stage model: %s", os.path.basename(gguf_path))
    return model
