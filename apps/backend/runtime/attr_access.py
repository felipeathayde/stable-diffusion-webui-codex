"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Attribute patching helpers for runtime modules.
Provides dotted-path getters/setters used by patchers and runtime glue code, including
non-inference parameter materialization for safe writable runtime reattachment.

Symbols (top-level; keep in sync; no ghosts):
- `_materialize_parameter_tensor` (function): Normalizes tensors/parameters into non-inference writable tensors for parameter wrapping.
- `set_attr` (function): Sets a nested attribute on an object by dotted path (type-aware).
- `set_attr_raw` (function): Sets a nested attribute by dotted path without conversions.
- `copy_to_param` (function): Copies a tensor/value into an existing `nn.Parameter` or tensor attribute.
- `get_attr` (function): Reads a nested attribute by dotted path.
- `get_attr_with_parent` (function): Reads a nested attribute and returns `(parent, attr_name, value)` for patching.
- `tensor2parameter` (function): Converts a tensor-like to an `nn.Parameter`.
"""

from __future__ import annotations

import torch


@torch.inference_mode(False)
@torch.no_grad()
def _materialize_parameter_tensor(value: torch.Tensor | torch.nn.Parameter) -> torch.Tensor:
    tensor = value.detach() if isinstance(value, torch.nn.Parameter) else value
    if torch.is_inference(tensor):
        tensor = tensor.clone()
    return tensor


def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], tensor2parameter(value))


def set_attr_raw(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)


def copy_to_param(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def get_attr_with_parent(obj, attr):
    attrs = attr.split(".")
    parent = obj
    name = None
    for name in attrs:
        parent = obj
        obj = getattr(obj, name)
    return parent, name, obj


def tensor2parameter(x):
    if isinstance(x, torch.nn.Parameter) and not x.requires_grad and not torch.is_inference(x):
        return x
    tensor = _materialize_parameter_tensor(x)
    parameter = torch.nn.Parameter(tensor, requires_grad=False)
    if torch.is_inference(parameter):
        raise RuntimeError("tensor2parameter produced inference parameter; materialization failed")
    return parameter


__all__ = [
    "copy_to_param",
    "get_attr",
    "get_attr_with_parent",
    "set_attr",
    "set_attr_raw",
    "tensor2parameter",
]
