"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF stage selection and model mounting.
Validates stage GGUF paths and mounts stage weights into `WanTransformer2DModel` via Codex GGUF operations (`using_codex_operations(weight_format="gguf")`) and WAN key remapping, with GGUF state loading/materialization wired to the memory-manager mount device (`dequantize` resolved from GGUF exec mode; `computation_dtype=dtype`) so placement policy remains centralized.
Optionally applies a per-stage LoRA file (merge/online) for LightX2V-style stage patches.

Symbols (top-level; keep in sync; no ghosts):
- `pick_stage_gguf` (function): Validates and returns the stage GGUF file path (strict: must be an explicit `.gguf` file).
- `_resolve_stage_mount_dequantize` (function): Resolves stage-mount GGUF dequant policy from runtime/env `gguf_exec` (`dequant_upfront` -> true; `dequant_forward` -> false; `cuda_pack` unsupported fail-loud).
- `_resolve_stage_mount_device` (function): Resolves the mount device from memory manager policy.
- `mount_stage_model_from_gguf` (function): Mounts a stage GGUF into a runtime transformer (mount-device GGUF load + key remapping + mount-device ops wrapper); final lifecycle ownership remains delegated to memory manager.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from apps.backend.infra.config.args import args as runtime_args
from apps.backend.infra.config.gguf_exec_mode import GgufExecMode, parse_gguf_exec_mode
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.checkpoint.io import load_gguf_state_dict

from .diagnostics import get_logger, log_cuda_mem
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


def _resolve_stage_mount_dequantize() -> bool:
    raw_mode = str(os.getenv("CODEX_GGUF_EXEC") or "").strip().lower()
    if not raw_mode:
        raw_mode = str(getattr(runtime_args, "gguf_exec", GgufExecMode.DEQUANT_FORWARD.value)).strip().lower()
    try:
        mode = parse_gguf_exec_mode(raw_mode)
    except ValueError as exc:
        raise RuntimeError(
            "WAN22 GGUF stage mount received invalid gguf exec mode: "
            f"{raw_mode!r} (expected dequant_forward or dequant_upfront)."
        ) from exc
    if mode == GgufExecMode.DEQUANT_UPFRONT:
        return True
    if mode == GgufExecMode.DEQUANT_FORWARD:
        return False
    raise RuntimeError(
        "WAN22 GGUF stage mount does not support gguf exec mode "
        f"{mode.value!r}. Use dequant_forward or dequant_upfront."
    )


def _resolve_stage_mount_device() -> torch.device:
    manager = getattr(memory_management, "manager", None)
    if manager is None or not hasattr(manager, "mount_device"):
        raise RuntimeError("WAN22 GGUF stage mount requires an active memory manager with mount_device().")
    mount_device = manager.mount_device()
    if not isinstance(mount_device, torch.device):
        raise RuntimeError(
            "WAN22 GGUF stage mount requires memory manager mount_device() to return torch.device "
            f"(got {type(mount_device).__name__})."
        )
    return mount_device


def mount_stage_model_from_gguf(
    gguf_path: str,
    *,
    stage: str,
    dtype: torch.dtype,
    lora_path: Optional[str] = None,
    lora_weight: Optional[float] = None,
    logger: Any,
):
    log = get_logger(logger)
    dequantize = _resolve_stage_mount_dequantize()
    mount_device = _resolve_stage_mount_device()
    log_cuda_mem(log, label=f"{stage}:before-mount-load")
    state = load_gguf_state_dict(
        gguf_path,
        dequantize=dequantize,
        computation_dtype=dtype,
        device=mount_device,
    )
    log_cuda_mem(log, label=f"{stage}:after-mount-load")
    state = remap_wan22_gguf_state_dict(state)
    with using_codex_operations(device=mount_device, dtype=dtype, weight_format="gguf"):
        model = load_wan_transformer_from_state_dict(state, config=None)
    del state
    log_cuda_mem(log, label=f"{stage}:after-mount-materialize")
    model.eval()
    apply_wan22_stage_lora(
        model,
        stage=stage,
        lora_path=lora_path,
        lora_weight=lora_weight,
        logger=logger,
    )
    log.info(
        "[wan22.gguf] mounted stage model: %s (dequantize=%s mount_device=%s)",
        os.path.basename(gguf_path),
        dequantize,
        mount_device,
    )
    return model
