from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Mapping, Tuple

import torch

from apps.backend.runtime.adapters.base import PatchKind, PatchSpec, log_missing_keys
from apps.backend.runtime.adapters.lora.types import (
    DiffWeights,
    GloraWeights,
    LohaWeights,
    LokrWeights,
    LoraWeights,
    SetWeights,
    make_spec,
)

LOGGER = logging.getLogger(__name__)


def _tensor_item(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    if value.numel() != 1:
        raise RuntimeError("Expected scalar tensor for alpha/dora_scale")
    return float(value.item())


def _maybe_convert_bfl_control(tensors: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "img_in.lora_A.weight" not in tensors or "single_blocks.0.norm.key_norm.scale" not in tensors:
        return dict(tensors)
    converted: Dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        new_key = key.replace(".lora_B.bias", ".diff_b").replace("_norm.scale", "_norm.scale.set_weight")
        converted[f"diffusion_model.{new_key}"] = value
    return converted


def _select_first_present(tensors: Mapping[str, torch.Tensor], names: Iterable[str]) -> Tuple[str | None, torch.Tensor | None]:
    for name in names:
        if name in tensors:
            return name, tensors[name]
    return None, None


def _extract_lora(
    logical_key: str,
    target_param: str,
    tensors: Mapping[str, torch.Tensor],
    loaded: set[str],
) -> PatchSpec | None:
    candidates = [
        (f"{logical_key}.lora_up.weight", f"{logical_key}.lora_down.weight", f"{logical_key}.lora_mid.weight"),
        (f"{logical_key}_lora.up.weight", f"{logical_key}_lora.down.weight", None),
        (f"{logical_key}.lora_B.weight", f"{logical_key}.lora_A.weight", None),
        (f"{logical_key}.lora.up.weight", f"{logical_key}.lora.down.weight", None),
        (f"{logical_key}.lora_linear_layer.up.weight", f"{logical_key}.lora_linear_layer.down.weight", None),
    ]
    A = B = mid = None
    a_name = b_name = m_name = None
    for up_name, down_name, mid_name in candidates:
        if up_name in tensors and down_name in tensors:
            a_name, b_name = up_name, down_name
            A, B = tensors[up_name], tensors[down_name]
            if mid_name and mid_name in tensors:
                m_name = mid_name
                mid = tensors[mid_name]
            break
    if A is None or B is None:
        return None

    alpha_tensor = tensors.get(f"{logical_key}.alpha")
    dora_scale = tensors.get(f"{logical_key}.dora_scale")

    if a_name:
        loaded.add(a_name)
    if b_name:
        loaded.add(b_name)
    if m_name:
        loaded.add(m_name)
    if alpha_tensor is not None:
        loaded.add(f"{logical_key}.alpha")
    if dora_scale is not None:
        loaded.add(f"{logical_key}.dora_scale")

    payload = LoraWeights(
        up=A,
        down=B,
        mid=mid,
        alpha=_tensor_item(alpha_tensor),
        dora_scale=dora_scale,
    )
    return make_spec(target_param, PatchKind.LORA, payload)


def _extract_loha(logical_key: str, target_param: str, tensors: Mapping[str, torch.Tensor], loaded: set[str]) -> PatchSpec | None:
    base = f"{logical_key}."
    required = ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"]
    if not all(f"{base}{name}" in tensors for name in required):
        return None
    w1_a = tensors[f"{base}hada_w1_a"]
    w1_b = tensors[f"{base}hada_w1_b"]
    w2_a = tensors[f"{base}hada_w2_a"]
    w2_b = tensors[f"{base}hada_w2_b"]
    t1 = tensors.get(f"{base}hada_t1")
    t2 = tensors.get(f"{base}hada_t2")
    alpha_tensor = tensors.get(f"{logical_key}.alpha")
    dora_scale = tensors.get(f"{logical_key}.dora_scale")
    for suffix in required:
        loaded.add(f"{base}{suffix}")
    for optional in ("hada_t1", "hada_t2"):
        name = f"{base}{optional}"
        if name in tensors:
            loaded.add(name)
    if alpha_tensor is not None:
        loaded.add(f"{logical_key}.alpha")
    if dora_scale is not None:
        loaded.add(f"{logical_key}.dora_scale")
    payload = LohaWeights(
        w1_a=w1_a,
        w1_b=w1_b,
        alpha=_tensor_item(alpha_tensor),
        w2_a=w2_a,
        w2_b=w2_b,
        t1=t1,
        t2=t2,
        dora_scale=dora_scale,
    )
    return make_spec(target_param, PatchKind.LOHA, payload)


def _extract_lokr(logical_key: str, target_param: str, tensors: Mapping[str, torch.Tensor], loaded: set[str]) -> PatchSpec | None:
    base = f"{logical_key}."
    keys = {
        "w1": tensors.get(f"{base}lokr_w1"),
        "w2": tensors.get(f"{base}lokr_w2"),
        "w1_a": tensors.get(f"{base}lokr_w1_a"),
        "w1_b": tensors.get(f"{base}lokr_w1_b"),
        "w2_a": tensors.get(f"{base}lokr_w2_a"),
        "w2_b": tensors.get(f"{base}lokr_w2_b"),
        "t2": tensors.get(f"{base}lokr_t2"),
    }
    if not any(value is not None for value in keys.values()):
        return None
    for suffix, value in keys.items():
        if value is not None:
            loaded.add(f"{base}lokr_{suffix}")
    alpha_tensor = tensors.get(f"{logical_key}.alpha")
    dora_scale = tensors.get(f"{logical_key}.dora_scale")
    if alpha_tensor is not None:
        loaded.add(f"{logical_key}.alpha")
    if dora_scale is not None:
        loaded.add(f"{logical_key}.dora_scale")
    payload = LokrWeights(
        w1=keys["w1"],
        w2=keys["w2"],
        alpha=_tensor_item(alpha_tensor),
        w1_a=keys["w1_a"],
        w1_b=keys["w1_b"],
        w2_a=keys["w2_a"],
        w2_b=keys["w2_b"],
        t2=keys["t2"],
        dora_scale=dora_scale,
    )
    return make_spec(target_param, PatchKind.LOKR, payload)


def _extract_glora(logical_key: str, target_param: str, tensors: Mapping[str, torch.Tensor], loaded: set[str]) -> PatchSpec | None:
    base = f"{logical_key}."
    required = ["a1.weight", "a2.weight", "b1.weight", "b2.weight"]
    if not all(f"{base}{name}" in tensors for name in required):
        return None
    a1 = tensors[f"{base}a1.weight"]
    a2 = tensors[f"{base}a2.weight"]
    b1 = tensors[f"{base}b1.weight"]
    b2 = tensors[f"{base}b2.weight"]
    alpha_tensor = tensors.get(f"{logical_key}.alpha")
    dora_scale = tensors.get(f"{logical_key}.dora_scale")
    for suffix in required:
        loaded.add(f"{base}{suffix}")
    if alpha_tensor is not None:
        loaded.add(f"{logical_key}.alpha")
    if dora_scale is not None:
        loaded.add(f"{logical_key}.dora_scale")
    payload = GloraWeights(
        a1=a1,
        a2=a2,
        b1=b1,
        b2=b2,
        alpha=_tensor_item(alpha_tensor),
        dora_scale=dora_scale,
    )
    return make_spec(target_param, PatchKind.GLORA, payload)


def _extract_diff(logical_key: str, target_param: str, tensors: Mapping[str, torch.Tensor], loaded: set[str]) -> List[PatchSpec]:
    specs: List[PatchSpec] = []
    diff = tensors.get(f"{logical_key}.diff")
    if diff is not None:
        loaded.add(f"{logical_key}.diff")
        specs.append(make_spec(target_param, PatchKind.DIFF, DiffWeights(weight=diff)))
    bias_key = f"{logical_key}.diff_b"
    if bias_key in tensors:
        if target_param.endswith(".weight"):
            bias_target = target_param[:-len(".weight")] + ".bias"
        else:
            bias_target = f"{target_param}.bias"
        specs.append(make_spec(bias_target, PatchKind.DIFF, DiffWeights(weight=tensors[bias_key])))
        loaded.add(bias_key)
    set_weight_key = f"{logical_key}.set_weight"
    if set_weight_key in tensors:
        specs.append(make_spec(target_param, PatchKind.SET, SetWeights(weight=tensors[set_weight_key])))
        loaded.add(set_weight_key)
    return specs


def parse_lora_tensors(tensors: Mapping[str, torch.Tensor], to_load: Dict[str, str]) -> tuple[List[PatchSpec], set[str]]:
    tensor_map = _maybe_convert_bfl_control(tensors)
    loaded: set[str] = set()
    specs: List[PatchSpec] = []

    for logical_key, target_param in to_load.items():
        extractor_sequence = (
            _extract_lora,
            _extract_loha,
            _extract_lokr,
            _extract_glora,
        )
        for extractor in extractor_sequence:
            spec = extractor(logical_key, target_param, tensor_map, loaded)
            if spec:
                specs.append(spec)
        specs.extend(_extract_diff(logical_key, target_param, tensor_map, loaded))

    log_missing_keys(tensor_map.keys(), loaded, logger=LOGGER)
    return specs, loaded
