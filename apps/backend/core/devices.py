"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Small device helpers for backend runtime code.
Provides canonical device helpers backed by memory-manager authority (`mount_device`/`cpu_device`) and best-effort
GC helpers to reduce VRAM fragmentation between runs.

Symbols (top-level; keep in sync; no ghosts):
- `_require_memory_manager` (function): Resolves the active memory-manager instance or fails loud when unavailable.
- `default_device` (function): Returns memory-manager mount device.
- `cpu` (function): Returns memory-manager CPU device.
- `torch_gc` (function): Best-effort cleanup (CUDA cache/IPCs + Python `gc.collect()`).
"""

from __future__ import annotations

import gc
import torch


def _require_memory_manager():
    from apps.backend.runtime.memory import memory_management

    manager = getattr(memory_management, "manager", None)
    if manager is None:
        raise RuntimeError("core.devices requires an active memory manager instance.")
    return manager


def default_device() -> torch.device:
    manager = _require_memory_manager()
    mount_device = manager.mount_device()
    if not isinstance(mount_device, torch.device):
        raise RuntimeError(
            "memory manager mount_device() must return torch.device "
            f"(got {type(mount_device).__name__})."
        )
    return mount_device


def cpu() -> torch.device:
    manager = _require_memory_manager()
    cpu_device = manager.cpu_device
    if not isinstance(cpu_device, torch.device):
        raise RuntimeError(
            "memory manager cpu_device must be torch.device "
            f"(got {type(cpu_device).__name__})."
        )
    return cpu_device


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
