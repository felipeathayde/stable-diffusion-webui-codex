"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA pipeline helpers for patch dict building and variant detection.
Converts parsed LoRA tensors into `ModelPatcher` patch dictionaries and provides a small helper to describe which adapter variants are present in a file.
Patch dictionary keys may be plain parameter names or `(parameter, offset)` tuples for slice patches.

Symbols (top-level; keep in sync; no ghosts):
- `VARIANT_LABELS` (constant): Mapping from `PatchKind` to stable string labels for variant reporting.
- `convert_specs_to_patch_dict` (function): Converts `PatchSpec` items into `ModelPatcher`-compatible patch dict tuples.
- `build_patch_dicts` (function): Builds patch dictionaries from raw tensors and a `to_load` key map.
- `describe_lora_file` (function): Returns the set of variant labels detected in a LoRA file.
"""

from __future__ import annotations

from typing import Dict, Mapping, Set

import torch

from apps.backend.runtime.adapters.base import PatchKind, PatchTarget
from apps.backend.runtime.adapters.lora.loader import parse_lora_tensors
from apps.backend.runtime.adapters.lora.types import (
    DiffWeights,
    GloraWeights,
    LohaWeights,
    LokrWeights,
    LoraWeights,
    SetWeights,
)


VARIANT_LABELS = {
    PatchKind.LORA: "lora",
    PatchKind.LOHA: "loha",
    PatchKind.LOKR: "lokr",
    PatchKind.GLORA: "glora",
    PatchKind.DIFF: "diff",
    PatchKind.SET: "set",
}


def convert_specs_to_patch_dict(specs) -> Dict[PatchTarget, tuple]:
    patch_dict: Dict[PatchTarget, tuple] = {}

    def _register_patch(parameter: PatchTarget, kind_label: str, weights: tuple[object, ...]) -> None:
        existing = patch_dict.get(parameter)
        if existing is not None:
            existing_kind, _ = existing
            raise RuntimeError(
                "LoRA patch collision: multiple variants target the same parameter "
                f"{parameter!r} (existing={existing_kind!r}, incoming={kind_label!r}). "
                "This is ambiguous and would otherwise silently overwrite adapter identity."
            )
        patch_dict[parameter] = (kind_label, weights)

    for spec in specs:
        payload = spec.payload
        if spec.kind == PatchKind.LORA:
            assert isinstance(payload, LoraWeights)
            _register_patch(
                spec.parameter,
                "lora",
                (
                    payload.up,
                    payload.down,
                    payload.alpha,
                    payload.mid,
                    payload.dora_scale,
                ),
            )
        elif spec.kind == PatchKind.LOHA:
            assert isinstance(payload, LohaWeights)
            _register_patch(
                spec.parameter,
                "loha",
                (
                    payload.w1_a,
                    payload.w1_b,
                    payload.alpha,
                    payload.w2_a,
                    payload.w2_b,
                    payload.t1,
                    payload.t2,
                    payload.dora_scale,
                ),
            )
        elif spec.kind == PatchKind.LOKR:
            assert isinstance(payload, LokrWeights)
            _register_patch(
                spec.parameter,
                "lokr",
                (
                    payload.w1,
                    payload.w2,
                    payload.alpha,
                    payload.w1_a,
                    payload.w1_b,
                    payload.w2_a,
                    payload.w2_b,
                    payload.t2,
                    payload.dora_scale,
                ),
            )
        elif spec.kind == PatchKind.GLORA:
            assert isinstance(payload, GloraWeights)
            _register_patch(
                spec.parameter,
                "glora",
                (
                    payload.a1,
                    payload.a2,
                    payload.b1,
                    payload.b2,
                    payload.alpha,
                    payload.dora_scale,
                ),
            )
        elif spec.kind == PatchKind.DIFF:
            assert isinstance(payload, DiffWeights)
            _register_patch(spec.parameter, "diff", (payload.weight,))
        elif spec.kind == PatchKind.SET:
            assert isinstance(payload, SetWeights)
            _register_patch(spec.parameter, "set", (payload.weight,))
        else:
            raise NotImplementedError(f"Unsupported LoRA patch kind: {spec.kind}")
    return patch_dict


def build_patch_dicts(tensors: Mapping[str, torch.Tensor], to_load: Dict[str, PatchTarget]) -> Dict[PatchTarget, tuple]:
    specs, _ = parse_lora_tensors(tensors, to_load)
    return convert_specs_to_patch_dict(specs)


def describe_lora_file(tensors: Mapping[str, torch.Tensor], to_load: Dict[str, PatchTarget]) -> Set[str]:
    patches, _ = parse_lora_tensors(tensors, to_load)
    return {VARIANT_LABELS[p.kind] for p in patches if p.kind in VARIANT_LABELS}
