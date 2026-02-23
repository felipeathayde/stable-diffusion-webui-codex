"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Quantization selection helpers for the GGUF converter.
Maps the human-facing quantization selector to GGML types and enforces generic per-tensor shape/block-size compatibility rules.

Symbols (top-level; keep in sync; no ghosts):
- `requested_ggml_type` (function): Maps `QuantizationType` to the requested `GGMLQuantizationType`.
- `select_tensor_ggml_type` (function): Selects the effective GGML type for a tensor given shape and requested type.
"""

from __future__ import annotations

from typing import Sequence

from apps.backend.quantization.gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from apps.backend.runtime.tools.gguf_converter_types import QuantizationType


def requested_ggml_type(quant: QuantizationType) -> GGMLQuantizationType:
    if quant == QuantizationType.F32:
        return GGMLQuantizationType.F32
    if quant == QuantizationType.F16:
        return GGMLQuantizationType.F16
    if quant == QuantizationType.Q8_0:
        return GGMLQuantizationType.Q8_0
    if quant == QuantizationType.Q5_K_M:
        return GGMLQuantizationType.Q5_K
    if quant == QuantizationType.Q6_K:
        return GGMLQuantizationType.Q6_K
    if quant == QuantizationType.Q5_K:
        return GGMLQuantizationType.Q5_K
    if quant == QuantizationType.Q5_1:
        return GGMLQuantizationType.Q5_1
    if quant == QuantizationType.Q5_0:
        return GGMLQuantizationType.Q5_0
    if quant == QuantizationType.Q4_K_M:
        return GGMLQuantizationType.Q4_K
    if quant == QuantizationType.Q4_K:
        return GGMLQuantizationType.Q4_K
    if quant == QuantizationType.Q4_1:
        return GGMLQuantizationType.Q4_1
    if quant == QuantizationType.Q4_0:
        return GGMLQuantizationType.Q4_0
    if quant == QuantizationType.Q3_K:
        return GGMLQuantizationType.Q3_K
    if quant == QuantizationType.Q2_K:
        return GGMLQuantizationType.Q2_K
    if quant == QuantizationType.IQ4_NL:
        return GGMLQuantizationType.IQ4_NL
    raise ValueError(f"Unsupported quantization: {quant}")


def select_tensor_ggml_type(shape: Sequence[int], requested: GGMLQuantizationType) -> GGMLQuantizationType:
    """Select the per-tensor GGML type.

    Behavior:
    - If requested is F16/BF16/F32: apply to all tensors.
    - Otherwise: keep 1D tensors in F16 and only quantize tensors whose last dim
      is divisible by the block size.
    """

    if requested in {GGMLQuantizationType.F16, GGMLQuantizationType.BF16, GGMLQuantizationType.F32}:
        return requested

    # Common GGUF convention: keep 1D tensors in F16.
    if len(shape) <= 1:
        return GGMLQuantizationType.F16

    block_size, _ = GGML_QUANT_SIZES[requested]
    if shape[-1] % block_size != 0:
        return GGMLQuantizationType.F16

    return requested


__all__ = [
    "requested_ggml_type",
    "select_tensor_ggml_type",
]
