"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Dataclasses describing LoRA-family weight payloads.
Defines typed payload structures for LoRA/LoHa/LoKr/GLoRA and related patch kinds, plus helpers for ordering and building `PatchSpec` entries.

Symbols (top-level; keep in sync; no ghosts):
- `LoraWeights` (dataclass): Payload for classic LoRA (up/down/mid/alpha/dora_scale).
- `LohaWeights` (dataclass): Payload for LoHa weights (w1/w2 components + optional Tucker tensors).
- `LokrWeights` (dataclass): Payload for LoKr weights (factorized weights + optional Tucker tensor).
- `GloraWeights` (dataclass): Payload for GLoRA weights (a/b matrices + alpha/dora_scale).
- `DiffWeights` (dataclass): Payload for diff-style patches.
- `SetWeights` (dataclass): Payload for set/overwrite patches.
- `LORA_VARIANT_ORDER` (constant): Stable ordering of patch kinds for deterministic processing.
- `make_spec` (function): Builds a `PatchSpec` from a patch-target/kind/payload triple (supports slice patch targets).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from apps.backend.runtime.adapters.base import PatchKind, PatchSpec, PatchTarget


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


def make_spec(parameter: PatchTarget, kind: PatchKind, payload: object) -> PatchSpec:
    return PatchSpec(parameter=parameter, kind=kind, payload=payload)
