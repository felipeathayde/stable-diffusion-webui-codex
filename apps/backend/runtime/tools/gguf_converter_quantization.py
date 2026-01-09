"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Quantization selection helpers for the GGUF converter.
Handles quantization presets, per-tensor override compilation, and shape/block-size compatibility rules.

Symbols (top-level; keep in sync; no ghosts):
- `requested_ggml_type` (function): Maps `QuantizationType` to the requested `GGMLQuantizationType`.
- `compile_tensor_overrides` (function): Compiles built-in + user-provided per-tensor quantization overrides.
- `select_tensor_ggml_type` (function): Selects the effective GGML type for a tensor given shape and requested type.
"""

from __future__ import annotations

import re
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


def _default_tensor_type_overrides(quant: QuantizationType) -> list[tuple[str, GGMLQuantizationType]]:
    """Return built-in per-tensor overrides for mixed-precision presets.

    Patterns are applied against both the source tensor name and the GGUF tensor name.
    """

    if quant == QuantizationType.Q5_K_M:
        return [
            # Embeddings / output: keep higher precision to preserve prompt semantics.
            (r"(?:^|\.)token_embd\.weight$", GGMLQuantizationType.Q8_0),
            (r"(?:^|\.)output\.weight$", GGMLQuantizationType.Q8_0),
            (r"model\.embed_tokens\.weight$", GGMLQuantizationType.Q8_0),
            (r"lm_head\.weight$", GGMLQuantizationType.Q8_0),
            # Attention projections: bump to 6-bit.
            (r"(?:^|\.)attn_(?:q|k|v|output)\.weight$", GGMLQuantizationType.Q6_K),
            (r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$", GGMLQuantizationType.Q6_K),
        ]

    if quant == QuantizationType.Q4_K_M:
        return [
            # Embeddings / output: bump to 6-bit (still much smaller than fp16).
            (r"(?:^|\.)token_embd\.weight$", GGMLQuantizationType.Q6_K),
            (r"(?:^|\.)output\.weight$", GGMLQuantizationType.Q6_K),
            (r"model\.embed_tokens\.weight$", GGMLQuantizationType.Q6_K),
            (r"lm_head\.weight$", GGMLQuantizationType.Q6_K),
            # Attention projections: bump to 5-bit K.
            (r"(?:^|\.)attn_(?:q|k|v|output)\.weight$", GGMLQuantizationType.Q5_K),
            (r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$", GGMLQuantizationType.Q5_K),
        ]

    return []


def compile_tensor_overrides(
    quant: QuantizationType,
    extra_overrides: Sequence[str],
) -> list[tuple[re.Pattern[str], GGMLQuantizationType]]:
    """Compile built-in + user-provided tensor quantization overrides.

    `extra_overrides` entries use a llama.cpp-like format: `<regex>=<quant>`
    where `<quant>` is any `QuantizationType` value (case-insensitive).
    """

    rules: list[tuple[re.Pattern[str], GGMLQuantizationType]] = []

    for pattern, qtype in _default_tensor_type_overrides(quant):
        rules.append((re.compile(pattern), qtype))

    for entry in extra_overrides:
        raw = str(entry or "").strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Invalid tensor override (expected '<regex>=<quant>'): {raw!r}")
        pattern, qname = raw.split("=", 1)
        pattern = pattern.strip()
        qname = qname.strip()
        if not pattern or not qname:
            raise ValueError(f"Invalid tensor override (expected '<regex>=<quant>'): {raw!r}")

        try:
            q_enum = QuantizationType(qname.upper())
        except ValueError as exc:
            raise ValueError(f"Invalid quant type in override {raw!r}: {qname!r}") from exc

        rules.append((re.compile(pattern), requested_ggml_type(q_enum)))

    return rules


def select_tensor_ggml_type(shape: Sequence[int], requested: GGMLQuantizationType) -> GGMLQuantizationType:
    """Select the per-tensor GGML type.

    Behavior:
    - If requested is F16/F32: apply to all tensors.
    - Otherwise: keep 1D tensors in F16 and only quantize tensors whose last dim
      is divisible by the block size.
    """

    if requested in {GGMLQuantizationType.F16, GGMLQuantizationType.F32}:
        return requested

    # Common GGUF convention: keep 1D tensors in F16.
    if len(shape) <= 1:
        return GGMLQuantizationType.F16

    block_size, _ = GGML_QUANT_SIZES[requested]
    if shape[-1] % block_size != 0:
        return GGMLQuantizationType.F16

    return requested


__all__ = [
    "compile_tensor_overrides",
    "requested_ggml_type",
    "select_tensor_ggml_type",
]

