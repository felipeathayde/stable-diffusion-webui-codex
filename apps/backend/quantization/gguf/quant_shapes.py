"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shape conversion helpers for quantized GGUF tensor storage.
Converts between logical tensor shapes (in elements) and packed byte shapes (in bytes) using `GGML_QUANT_SIZES` for a given quantization type.

Symbols (top-level; keep in sync; no ghosts):
- `quant_shape_to_byte_shape` (function): Converts a logical tensor shape into its packed byte shape for a quant type.
- `quant_shape_from_byte_shape` (function): Converts a packed byte shape into the logical tensor shape for a quant type.
"""

from __future__ import annotations

from collections.abc import Sequence

from .constants import GGML_QUANT_SIZES, GGMLQuantizationType


def quant_shape_to_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(
            f"Quantized tensor row size ({shape[-1]}) is not a multiple of {quant_type.name} block size ({block_size})"
        )
    return (*shape[:-1], shape[-1] // block_size * type_size)


def quant_shape_from_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % type_size != 0:
        raise ValueError(
            f"Quantized tensor bytes per row ({shape[-1]}) is not a multiple of {quant_type.name} type size ({type_size})"
        )
    return (*shape[:-1], shape[-1] // type_size * block_size)
