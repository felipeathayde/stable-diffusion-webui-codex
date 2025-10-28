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
