"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared tensor-tree helpers for engine caching.
Centralizes “detach+to(cpu)” and “move to device/dtype” logic used by multiple engines when storing/restoring cached
conditioning payloads.

Symbols (top-level; keep in sync; no ghosts):
- `detach_to_cpu` (function): Recursively detach tensors and move them to CPU (dict/tuple/list supported).
- `move_to_device` (function): Recursively move tensors to a target device (and optional dtype).
"""

from __future__ import annotations

from typing import Any

import torch


def detach_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {k: detach_to_cpu(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(v) for v in value)
    if isinstance(value, list):
        return [detach_to_cpu(v) for v in value]
    return value


def move_to_device(
    value: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> Any:
    if isinstance(value, torch.Tensor):
        if dtype is None:
            return value.to(device)
        return value.to(device=device, dtype=dtype)
    if isinstance(value, dict):
        return {k: move_to_device(v, device=device, dtype=dtype) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device=device, dtype=dtype) for v in value)
    if isinstance(value, list):
        return [move_to_device(v, device=device, dtype=dtype) for v in value]
    return value

