"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tensor planning helpers for the GGUF converter.
Plans tensor name remaps, quantization types, and storage byte shapes without loading full tensors.

Symbols (top-level; keep in sync; no ghosts):
- `TensorPlan` (dataclass): Planned tensor conversion entry (name/shape/type + storage strategy).
- `plan_tensors` (function): Plan per-tensor conversion settings for a safetensors source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from apps.backend.quantization.gguf import GGMLQuantizationType
from apps.backend.quantization.gguf.quant_shapes import quant_shape_to_byte_shape
from apps.backend.runtime.tools.gguf_converter_quantization import select_tensor_ggml_type


@dataclass(frozen=True, slots=True)
class TensorPlan:
    src_name: str
    gguf_name: str
    raw_shape: tuple[int, ...]
    ggml_type: GGMLQuantizationType
    stored_shape: tuple[int, ...]
    stored_dtype: np.dtype
    stored_nbytes: int


def plan_tensors(
    tensor_names: list[str],
    safetensors_handle: Any,
    key_mapping: dict[str, str],
    requested: GGMLQuantizationType,
    overrides: list[tuple[re.Pattern[str], GGMLQuantizationType]],
) -> list[TensorPlan]:
    plans: list[TensorPlan] = []

    for src_name in tensor_names:
        sl = safetensors_handle.get_slice(src_name)
        raw_shape = tuple(int(x) for x in sl.get_shape())
        gguf_name = key_mapping.get(src_name, src_name)

        desired = requested
        for rx, qtype in overrides:
            if rx.search(src_name) or rx.search(gguf_name):
                desired = qtype
        ggml_type = select_tensor_ggml_type(raw_shape, desired)

        if ggml_type == GGMLQuantizationType.F16:
            stored_dtype = np.dtype(np.float16)
            stored_shape = raw_shape
            stored_nbytes = int(np.prod(raw_shape, dtype=np.int64) * 2)
        elif ggml_type == GGMLQuantizationType.F32:
            stored_dtype = np.dtype(np.float32)
            stored_shape = raw_shape
            stored_nbytes = int(np.prod(raw_shape, dtype=np.int64) * 4)
        else:
            stored_dtype = np.dtype(np.uint8)
            stored_shape = quant_shape_to_byte_shape(raw_shape, ggml_type)
            stored_nbytes = int(np.prod(stored_shape, dtype=np.int64))

        plans.append(
            TensorPlan(
                src_name=src_name,
                gguf_name=gguf_name,
                raw_shape=raw_shape,
                ggml_type=ggml_type,
                stored_shape=stored_shape,
                stored_dtype=stored_dtype,
                stored_nbytes=stored_nbytes,
            )
        )

    return plans


__all__ = [
    "TensorPlan",
    "plan_tensors",
]

