"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Autocast guard helpers for runtime fp32 compute paths.
Provides a safe way to disable torch AMP autocast for devices that support it, while remaining a no-op for device types
that do not (e.g. DirectML `device.type == "dml"`).

Symbols (top-level; keep in sync; no ghosts):
- `device_type_supports_autocast` (function): Returns whether `torch.amp.autocast(device_type=...)` is supported.
- `autocast_disabled` (function): Context manager that disables autocast when supported; otherwise a no-op.
"""

from __future__ import annotations

import contextlib
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def device_type_supports_autocast(device_type: str) -> bool:
    try:
        torch.amp.autocast(device_type=str(device_type), enabled=False)
    except Exception:
        return False
    return True


@contextlib.contextmanager
def autocast_disabled(device_type: str):
    device_type = str(device_type)
    if device_type_supports_autocast(device_type):
        with torch.amp.autocast(device_type=device_type, enabled=False):
            yield
        return
    yield


__all__ = [
    "autocast_disabled",
    "device_type_supports_autocast",
]

