"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Small device helpers for backend runtime code.
Provides a default CUDA/CPU device chooser and best-effort GC helpers to reduce VRAM fragmentation between runs.

Symbols (top-level; keep in sync; no ghosts):
- `default_device` (function): Returns CUDA device if available, otherwise CPU.
- `cpu` (function): Returns a CPU `torch.device`.
- `torch_gc` (function): Best-effort cleanup (CUDA cache/IPCs + Python `gc.collect()`).
"""

from __future__ import annotations

import gc
import torch


def default_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def cpu() -> torch.device:
    return torch.device("cpu")


def torch_gc() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


__all__ = ["default_device", "cpu", "torch_gc"]
