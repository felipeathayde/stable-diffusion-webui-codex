"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared LoRA patch types for the patcher subsystem.
Defines the normalized patch segment representation used by the merge/applier helpers.

Symbols (top-level; keep in sync; no ghosts):
- `LoraPatchEntry` (type): Raw LoRA patch entry tuple/list shape used by conversion helpers.
- `LoraVariant` (enum): Supported LoRA patch variants (diff/set/lora/loha/lokr/glora) with tag parsing.
- `OffsetSpec` (dataclass): Tensor narrow/slice spec used for offset-based LoRA segments.
- `LoraPatchSegment` (dataclass): Normalized patch segment representation (variant + tensors + offsets) used by apply helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence

import torch

LoraPatchEntry = Sequence[object]


class LoraVariant(Enum):
    DIFF = "diff"
    SET = "set"
    LORA = "lora"
    LOHA = "loha"
    LOKR = "lokr"
    GLORA = "glora"

    @classmethod
    def from_tag(cls, tag: str) -> "LoraVariant":
        for variant in cls:
            if variant.value == tag:
                return variant
        raise ValueError(f"Unsupported LoRA patch type '{tag}'")


@dataclass(frozen=True)
class OffsetSpec:
    dim: int
    start: int
    length: int

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.narrow(self.dim, self.start, self.length)


@dataclass(frozen=True)
class LoraPatchSegment:
    strength_patch: float
    strength_model: float
    payload: object
    offset: Optional[OffsetSpec]
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]]
    variant: Optional[LoraVariant]
    custom_kind: Optional[str]


__all__ = [
    "LoraPatchEntry",
    "LoraPatchSegment",
    "LoraVariant",
    "OffsetSpec",
]

