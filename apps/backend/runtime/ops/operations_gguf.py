"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GGUF runtime operations backed by `apps.backend.quantization` (CodexQuantization).
Provides direct dequantization helpers (`dequantize_tensor*`) with fail-loud handling for removed dequant-cache policies.

Symbols (top-level; keep in sync; no ghosts):
- `CodexParameter` (class): Packed GGUF tensor wrapper (imported from `apps.backend.quantization.tensor`).
- `CodexPackLinearQ4KTilepackV1Parameter` (class): Packed CodexPack linear-weight wrapper (imported from `apps.backend.quantization.codexpack_tensor`).
- `set_cache_policy` (function): Validates removed dequant-cache settings (only disabled policy accepted).
- `clear_cache` (function): No-op retained for cache-clear callsites after dequant-cache removal.
- `dequantize_tensor_for_forward` (function): Dequantize a GGUF tensor for forward on target device/dtype (no run-scoped cache).
- `dequantize_tensor` (function): Dequantize a `CodexParameter` to a float tensor (pass-through for non-GGUF tensors).
- `__all__` (constant): Public export list for GGUF runtime operations.
"""

from __future__ import annotations

import logging

import torch

from apps.backend.quantization.api import dequantize as codex_dequantize
from apps.backend.quantization.codexpack_tensor import CodexPackLinearQ4KTilepackV1Parameter
from apps.backend.quantization.tensor import CodexParameter

__all__ = [
    "CodexPackLinearQ4KTilepackV1Parameter",
    "CodexParameter",
    "dequantize_tensor_for_forward",
    "dequantize_tensor",
    "set_cache_policy",
    "clear_cache",
]

_LOG = logging.getLogger("backend.runtime.ops.operations_gguf")


def set_cache_policy(policy: str = "none", limit_mb: int = 0) -> None:
    """Validate GGUF dequant cache policy after cache removal.

    Only disabled cache policies are accepted. Any non-disabled request fails loud.
    """

    normalized = (policy or "none").strip().lower()
    disabled = {"", "none", "off"}
    if normalized not in disabled:
        raise RuntimeError(
            "GGUF dequant cache was removed. "
            f"Unsupported policy={policy!r}; use gguf_cache_policy='none' (or omit it)."
        )
    normalized_limit_mb = int(limit_mb or 0)
    if normalized_limit_mb != 0:
        raise RuntimeError(
            "GGUF dequant cache was removed. "
            f"gguf_cache_limit_mb must be 0 or omitted (got {normalized_limit_mb})."
        )
    _LOG.debug("GGUF dequant cache disabled (policy=%r limit_mb=%d).", policy, normalized_limit_mb)


def clear_cache() -> None:
    """No-op retained for callers that clear caches between stages."""

    _LOG.debug("GGUF dequant cache clear requested after removal (no-op).")


def dequantize_tensor(tensor):
    """Return a dequantized float tensor (or pass-through for non-quant tensors)."""
    if tensor is None:
        return None
    if not isinstance(tensor, CodexParameter) or tensor.qtype is None:
        return tensor
    return codex_dequantize(tensor)


def dequantize_tensor_for_forward(
    tensor: torch.Tensor | None,
    *,
    target_device: torch.device,
    target_dtype: torch.dtype | None,
    non_blocking: bool = False,
) -> torch.Tensor | None:
    """Dequantize a GGUF tensor for forward on the requested device/dtype (no run-scoped cache)."""

    if tensor is None:
        return None
    if not isinstance(tensor, CodexParameter) or tensor.qtype is None:
        return tensor
    moved = tensor.to(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
    return codex_dequantize(moved)
