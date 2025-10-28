from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from apps.backend.runtime.adapters.base import PatchKind, PatchSpec


@dataclass(frozen=True)
class LoraWeights:
    up: torch.Tensor
    down: torch.Tensor
    mid: torch.Tensor | None
    alpha: Optional[float]
    dora_scale: torch.Tensor | None


@dataclass(frozen=True)
class LohaWeights:
    w1_a: torch.Tensor
    w1_b: torch.Tensor
    alpha: Optional[float]
    w2_a: torch.Tensor
    w2_b: torch.Tensor
    t1: torch.Tensor | None
    t2: torch.Tensor | None
    dora_scale: torch.Tensor | None


@dataclass(frozen=True)
class LokrWeights:
    w1: torch.Tensor | None
    w2: torch.Tensor | None
    alpha: Optional[float]
    w1_a: torch.Tensor | None
    w1_b: torch.Tensor | None
    w2_a: torch.Tensor | None
    w2_b: torch.Tensor | None
    t2: torch.Tensor | None
    dora_scale: torch.Tensor | None


@dataclass(frozen=True)
class GloraWeights:
    a1: torch.Tensor
    a2: torch.Tensor
    b1: torch.Tensor
    b2: torch.Tensor
    alpha: Optional[float]
    dora_scale: torch.Tensor | None


@dataclass(frozen=True)
class DiffWeights:
    weight: torch.Tensor


@dataclass(frozen=True)
class SetWeights:
    weight: torch.Tensor


LORA_VARIANT_ORDER = (
    PatchKind.LORA,
    PatchKind.LOHA,
    PatchKind.LOKR,
    PatchKind.GLORA,
    PatchKind.DIFF,
    PatchKind.SET,
)


def make_spec(parameter: str, kind: PatchKind, payload: object) -> PatchSpec:
    return PatchSpec(parameter=parameter, kind=kind, payload=payload)
