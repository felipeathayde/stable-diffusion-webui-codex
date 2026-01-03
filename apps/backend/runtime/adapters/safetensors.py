"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SafeTensors helper wrappers for runtime adapter loading.
Provides a small exception type and safe-open context manager used by adapter loaders to read tensors without leaking handles.

Symbols (top-level; keep in sync; no ghosts):
- `SafeTensorError` (class): Adapter-facing error for safetensors open/load failures.
- `open_safetensor` (contextmanager): Context manager opening a safetensors file and ensuring the handle is closed.
- `load_tensors` (function): Loads all tensors from a safetensors file into a `{key: tensor}` mapping.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

import safetensors
import safetensors.torch as sf
import torch


class SafeTensorError(RuntimeError):
    pass


@contextmanager
def open_safetensor(path: str) -> Iterator[sf.SafeTensor]:
    try:
        handle = sf.safe_open(path, framework="pt")
    except (safetensors.SafetensorError, FileNotFoundError) as exc:
        raise SafeTensorError(f"Unable to open safetensor '{path}': {exc}") from exc
    try:
        yield handle
    finally:
        handle.close()


def load_tensors(path: str) -> Mapping[str, torch.Tensor]:
    with open_safetensor(path) as handle:
        keys = list(handle.keys())
        tensors = {key: handle.get_tensor(key) for key in keys}
    return tensors
