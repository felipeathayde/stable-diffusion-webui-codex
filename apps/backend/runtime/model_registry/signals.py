"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Lazy state-dict signal extraction helpers for model detection.
Wraps a checkpoint mapping in a `SignalBundle` that exposes keys and lazily computed shapes, plus small helpers used across detectors.

Symbols (top-level; keep in sync; no ghosts):
- `SignalBundle` (dataclass): State-dict wrapper exposing keys and lazy/cached shape lookup.
- `build_bundle` (function): Builds a `SignalBundle` without materializing all tensors.
- `count_blocks` (function): Counts sequential blocks matching a template prefix pattern.
- `has_all_keys` (function): Returns True if all required keys exist in a bundle.
- `get_tensor_dtype` (function): Best-effort dtype name extraction for a tensor-like object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Tuple


@dataclass
class SignalBundle:
    state_dict: Mapping[str, Any]
    keys: Tuple[str, ...]
    shapes: MutableMapping[str, Tuple[int, ...]]

    def shape(self, key: str) -> Tuple[int, ...] | None:
        # Fast path: cached
        cached = self.shapes.get(key)
        if cached is not None:
            return cached
        # Lazy path: load a single tensor from the mapping (avoids materializing thousands of tensors)
        try:
            v = self.state_dict[key]
        except Exception:
            return None
        shape = getattr(v, "shape", None)
        if shape is None and hasattr(v, "size"):
            try:
                shape = tuple(int(x) for x in v.size())  # type: ignore[arg-type]
            except Exception:
                shape = None
        if shape is None:
            return None
        try:
            # Cache for subsequent calls when shapes is mutable
            self.shapes[key] = tuple(int(x) for x in shape)  # type: ignore[index]
        except Exception:
            pass
        return tuple(int(x) for x in shape)

    def has_prefix(self, prefix: str) -> bool:
        return any(k.startswith(prefix) for k in self.keys)


def build_bundle(state_dict: Mapping[str, Any]) -> SignalBundle:
    # Do not materialize all tensors; only list keys and compute shapes lazily via SignalBundle.shape()
    keys = tuple(state_dict.keys())
    shapes: dict[str, Tuple[int, ...]] = {}
    return SignalBundle(state_dict=state_dict, keys=keys, shapes=shapes)


def count_blocks(keys: Iterable[str], template: str) -> int:
    """Count sequential blocks in keys following a zero-based template.

    Example template: ``"model.diffusion_model.input_blocks.{}."``
    """
    count = 0
    while True:
        prefix = template.format(count)
        if not any(k.startswith(prefix) for k in keys):
            break
        count += 1
    return count


def has_all_keys(bundle: SignalBundle, *required: str) -> bool:
    return all(k in bundle.state_dict for k in required)


def get_tensor_dtype(tensor: Any) -> str | None:
    return getattr(getattr(tensor, "dtype", None), "name", None)
