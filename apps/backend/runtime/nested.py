"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Nested structure helpers for runtime codepaths.
Provides recursive helpers to size and move nested structures that contain tensors and parameters.

Symbols (top-level; keep in sync; no ghosts):
- `fp16_fix` (function): Applies fp16 compatibility fixes for legacy checkpoints (best-effort).
- `dtype_to_element_size` (function): Returns element size in bytes for a dtype name/torch dtype.
- `nested_compute_size` (function): Computes total size of a nested object tree (dict/list/tuples) given element size.
- `nested_move_to_device` (function): Recursively moves tensors/parameters in a nested structure to a device/dtype.
"""

from __future__ import annotations

import torch


def fp16_fix(x):
    # Avoid fp16 overflow (Diffusers-style clamp).
    if x.dtype in [torch.float16]:
        return x.clip(-32768.0, 32768.0)
    return x


def dtype_to_element_size(dtype):
    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).element_size()
    raise ValueError(f"Invalid dtype: {dtype}")


def nested_compute_size(obj, element_size):
    module_mem = 0

    if isinstance(obj, dict):
        for key in obj:
            module_mem += nested_compute_size(obj[key], element_size)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for i in range(len(obj)):
            module_mem += nested_compute_size(obj[i], element_size)
    elif isinstance(obj, torch.Tensor):
        module_mem += obj.nelement() * element_size

    return module_mem


def nested_move_to_device(obj, **kwargs):
    if isinstance(obj, dict):
        for key in obj:
            obj[key] = nested_move_to_device(obj[key], **kwargs)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = nested_move_to_device(obj[i], **kwargs)
    elif isinstance(obj, tuple):
        obj = tuple(nested_move_to_device(i, **kwargs) for i in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.to(**kwargs)
    return obj


__all__ = [
    "dtype_to_element_size",
    "fp16_fix",
    "nested_compute_size",
    "nested_move_to_device",
]

