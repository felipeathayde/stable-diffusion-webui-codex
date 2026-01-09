"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Verification helpers for the GGUF converter.
Validates written GGUF metadata/tensor tables and runs cheap spot-checks against source safetensors.

Symbols (top-level; keep in sync; no ghosts):
- `verify_gguf_file` (function): Verifies a written GGUF file and raises on mismatch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from apps.backend.quantization.api import dequantize_numpy
from apps.backend.quantization.gguf import GGMLQuantizationType, GGUFReader
from apps.backend.runtime.tools import gguf_converter_safetensors_source as _safetensors_source
from apps.backend.runtime.tools.gguf_converter_tensor_planner import TensorPlan
from apps.backend.runtime.tools.gguf_converter_types import GGUFVerificationError


def verify_gguf_file(
    gguf_path: str,
    source_safetensors: str,
    *,
    tensor_plans: list[TensorPlan],
    key_mapping: dict[str, str],
) -> None:
    gguf_path_path = Path(gguf_path)
    if not gguf_path_path.exists():
        raise GGUFVerificationError(f"GGUF file does not exist: {gguf_path_path}")

    # 1) Parse with the repo GGUF reader (validates header/KV/TI/offsets).
    reader = GGUFReader(str(gguf_path_path))

    expected_count = len(tensor_plans)
    if len(reader.tensors) != expected_count:
        raise GGUFVerificationError(
            f"Tensor count mismatch: GGUF has {len(reader.tensors)}, expected {expected_count}"
        )

    by_name = {t.name: t for t in reader.tensors}

    for plan in tensor_plans:
        if plan.gguf_name not in by_name:
            raise GGUFVerificationError(f"Tensor missing in GGUF: {plan.gguf_name}")
        t = by_name[plan.gguf_name]
        if t.tensor_type != plan.ggml_type:
            raise GGUFVerificationError(
                f"DTYPE mismatch for {plan.gguf_name}: GGUF has {t.tensor_type.name}, expected {plan.ggml_type.name}"
            )
        expected_shape_gguf = tuple(reversed(plan.raw_shape))
        if tuple(int(x) for x in t.shape) != expected_shape_gguf:
            raise GGUFVerificationError(
                f"Shape mismatch for {plan.gguf_name}: GGUF has {tuple(int(x) for x in t.shape)}, expected {expected_shape_gguf}"
            )
        if int(t.n_bytes) != int(plan.stored_nbytes):
            raise GGUFVerificationError(
                f"Byte size mismatch for {plan.gguf_name}: GGUF has {int(t.n_bytes)}, expected {int(plan.stored_nbytes)}"
            )

    def _quant_spotcheck_tol(qtype: GGMLQuantizationType) -> tuple[float, float]:
        # Spot-check tolerances are intentionally loose: goal is catching layout/packing bugs,
        # not measuring perceptual quality.
        if qtype == GGMLQuantizationType.Q2_K:
            return (1.0, 1.0)
        if qtype == GGMLQuantizationType.Q3_K:
            return (0.8, 0.8)
        return (0.6, 0.6)

    # 2) Spot-check a few tensors against source.
    reverse_mapping = {v: k for k, v in key_mapping.items()}
    with _safetensors_source.open_safetensors_source(source_safetensors) as source:
        for plan in tensor_plans[:3]:
            src_name = reverse_mapping.get(plan.gguf_name, plan.gguf_name)
            if src_name not in source.keys():
                continue

            gguf_tensor = by_name[plan.gguf_name]
            src_slice = source.get_slice(src_name)
            src_shape = tuple(int(x) for x in src_slice.get_shape())

            if plan.ggml_type == GGMLQuantizationType.F16:
                if len(src_shape) == 1:
                    src = src_slice[:4].to(torch.float16).flatten().numpy().tobytes(order="C")
                else:
                    src = src_slice[:1].to(torch.float16).flatten()[:4].numpy().tobytes(order="C")
                gg = gguf_tensor.data.reshape(-1).view(np.uint8)[: len(src)].tobytes(order="C")
                if gg != src:
                    raise GGUFVerificationError(f"Tensor data mismatch (F16) for {plan.gguf_name}")

            elif plan.ggml_type == GGMLQuantizationType.F32:
                if len(src_shape) == 1:
                    src = src_slice[:4].to(torch.float32).flatten().numpy().tobytes(order="C")
                else:
                    src = src_slice[:1].to(torch.float32).flatten()[:4].numpy().tobytes(order="C")
                gg = gguf_tensor.data.reshape(-1).view(np.uint8)[: len(src)].tobytes(order="C")
                if gg != src:
                    raise GGUFVerificationError(f"Tensor data mismatch (F32) for {plan.gguf_name}")

            else:
                # Quantized: dequantize first row and compare roughly.
                row_bytes = gguf_tensor.data.reshape((-1, gguf_tensor.data.shape[-1]))[0]
                out = dequantize_numpy(row_bytes, plan.ggml_type)

                # Avoid loading the full tensor: grab only the first outer slice.
                ref_chunk = src_slice[:1].reshape(-1, src_shape[-1])[0].float().numpy()
                n = min(256, ref_chunk.shape[0])
                if not np.all(np.isfinite(out[:n])):
                    raise GGUFVerificationError(f"Non-finite dequant output for {plan.gguf_name}")
                rtol, atol = _quant_spotcheck_tol(plan.ggml_type)
                if not np.allclose(out[:n], ref_chunk[:n], rtol=rtol, atol=atol):
                    raise GGUFVerificationError(f"Quantized spot-check mismatch for {plan.gguf_name}")


__all__ = [
    "verify_gguf_file",
]

