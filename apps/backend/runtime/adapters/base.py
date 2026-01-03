"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared adapter primitives and small utilities for patch application.
Defines patch kinds/specs and provides helpers for tensor casting/shape coercion and strict validation used by the native adapter pipeline.

Symbols (top-level; keep in sync; no ghosts):
- `PatchKind` (enum): Patch variant identifiers (LoRA/LoHa/LoKr/GLoRA/Diff/Set).
- `PatchSpec` (dataclass): Patch record ready to be consumed by `ModelPatcher` (parameter + kind + payload).
- `cast_tensor` (function): Casts a tensor to a target device/dtype using runtime memory management.
- `ensure_same_shape` (function): Reshapes a tensor to match a target shape when needed.
- `log_missing_keys` (function): Logs LoRA keys that were ignored/missing during mapping.
- `require` (function): Raises `RuntimeError` when a condition is not met (fail-fast helper).
- `ensure_not_implemented` (function): Raises `NotImplementedError` for unported features in the native adapter stack.
- `tensor_eps` (function): Returns a dtype/device-matched epsilon tensor for the given tensor dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable

import torch

from apps.backend.runtime.memory import memory_management


class PatchKind(Enum):
    LORA = auto()
    LOHA = auto()
    LOKR = auto()
    GLORA = auto()
    DIFF = auto()
    SET = auto()


@dataclass(frozen=True)
class PatchSpec:
    """Patch ready to be consumed by ModelPatcher."""

    parameter: str
    kind: PatchKind
    payload: object


def cast_tensor(tensor: torch.Tensor, device: torch.device | None, dtype: torch.dtype | None) -> torch.Tensor:
    """Move tensor to the requested device / dtype."""
    target_device = device or tensor.device
    target_dtype = dtype or tensor.dtype
    return memory_management.cast_to_device(tensor, target_device, target_dtype)


def ensure_same_shape(tensor: torch.Tensor, shape: Iterable[int]) -> torch.Tensor:
    """Reshape tensor if needed to match a target shape."""
    if list(tensor.shape) == list(shape):
        return tensor
    return tensor.reshape(*shape)


def log_missing_keys(all_keys: Iterable[str], loaded_keys: Iterable[str], *, logger) -> None:
    missing = sorted(set(all_keys) - set(loaded_keys))
    for key in missing:
        logger.debug("LoRA key ignored: %s", key)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def ensure_not_implemented(feature: str) -> None:
    raise NotImplementedError(f"{feature} is not implemented in the native adapter stack.")


def tensor_eps(tensor: torch.Tensor) -> torch.Tensor:
    return torch.full((1,), torch.finfo(tensor.dtype).eps, dtype=tensor.dtype, device=tensor.device)
