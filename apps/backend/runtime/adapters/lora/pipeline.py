from __future__ import annotations

from typing import Dict, Iterable, Mapping, Set

import torch

from apps.backend.runtime.adapters.base import PatchKind
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


def convert_specs_to_patch_dict(specs) -> Dict[str, tuple]:
    patch_dict: Dict[str, tuple] = {}
    for spec in specs:
        payload = spec.payload
        if spec.kind == PatchKind.LORA:
            assert isinstance(payload, LoraWeights)
            patch_dict[spec.parameter] = (
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
            patch_dict[spec.parameter] = (
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
            patch_dict[spec.parameter] = (
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
            patch_dict[spec.parameter] = (
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
            patch_dict[spec.parameter] = ("diff", (payload.weight,))
        elif spec.kind == PatchKind.SET:
            assert isinstance(payload, SetWeights)
            patch_dict[spec.parameter] = ("set", (payload.weight,))
        else:
            raise NotImplementedError(f"Unsupported LoRA patch kind: {spec.kind}")
    return patch_dict


def build_patch_dicts(tensors: Mapping[str, torch.Tensor], to_load: Dict[str, str]) -> Dict[str, tuple]:
    specs, _ = parse_lora_tensors(tensors, to_load)
    return convert_specs_to_patch_dict(specs)


def describe_lora_file(tensors: Mapping[str, torch.Tensor], to_load: Dict[str, str]) -> Set[str]:
    patches, _ = parse_lora_tensors(tensors, to_load)
    return {VARIANT_LABELS[p.kind] for p in patches if p.kind in VARIANT_LABELS}
