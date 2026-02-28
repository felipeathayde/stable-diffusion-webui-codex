"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GGUF runtime operations backed by `apps.backend.quantization` (CodexQuantization).
Provides direct dequantization helpers for runtime GGUF parameters.

Symbols (top-level; keep in sync; no ghosts):
- `CodexParameter` (class): Packed GGUF tensor wrapper (imported from `apps.backend.quantization.tensor`).
- `CodexPackLinearQ4KTilepackV1Parameter` (class): Packed CodexPack linear-weight wrapper (imported from `apps.backend.quantization.codexpack_tensor`).
- `dequantize_tensor` (function): Dequantize a `CodexParameter` to a float tensor (pass-through for non-GGUF tensors).
- `__all__` (constant): Public export list for GGUF runtime operations.
"""

from __future__ import annotations

from apps.backend.quantization.api import dequantize as codex_dequantize
from apps.backend.quantization.codexpack_tensor import CodexPackLinearQ4KTilepackV1Parameter
from apps.backend.quantization.tensor import CodexParameter

__all__ = [
    "CodexPackLinearQ4KTilepackV1Parameter",
    "CodexParameter",
    "dequantize_tensor",
]


def dequantize_tensor(tensor):
    """Return a dequantized float tensor (or pass-through for non-quant tensors)."""
    if tensor is None:
        return None
    if not isinstance(tensor, CodexParameter) or tensor.qtype is None:
        return tensor
    return codex_dequantize(tensor)
