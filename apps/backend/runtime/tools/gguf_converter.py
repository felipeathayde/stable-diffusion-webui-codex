"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GGUF converter tool (SafeTensors → GGUF) with optional quantization, metadata injection, and verification.
Used primarily for text encoders and other components that need GGUF artifacts (e.g. ZImage Qwen3 variants), including sharded HF-style weights.

Symbols (top-level; keep in sync; no ghosts):
- `QuantizationType` (enum): Supported “human” quantization selectors for conversion (maps to `GGMLQuantizationType`).
- `ConversionConfig` (dataclass): Conversion configuration (input/output paths, quantization choices, tensor overrides, and metadata inputs).
- `ConversionProgress` (dataclass): Progress/report structure for long conversions (stage counters, timings, and status fields).
- `GGUFConversionCancelled` (exception): Raised when a conversion is cancelled via a cooperative cancel signal.
- `GGUFVerificationError` (exception): Raised when a written GGUF file fails validation/verification.
- `convert_safetensors_to_gguf` (function): Main conversion entrypoint; reads SafeTensors (incl. sharded), quantizes tensors, writes GGUF,
  and optionally verifies the output (uses many helpers above).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import torch

from apps.backend.quantization.api import quantize_numpy
from apps.backend.quantization.gguf import (
    GGMLQuantizationType,
    GGUFWriter,
)
from apps.backend.runtime.tools import gguf_converter_metadata as _metadata
from apps.backend.runtime.tools import gguf_converter_profiles as _profiles
from apps.backend.runtime.tools import gguf_converter_quantization as _quantization
from apps.backend.runtime.tools import gguf_converter_safetensors_source as _safetensors_source
from apps.backend.runtime.tools import gguf_converter_tensor_planner as _tensor_planner
from apps.backend.runtime.tools import gguf_converter_verify as _verify
from apps.backend.runtime.tools.gguf_converter_specs import GGUFArch, GGUFKeyLayout
from apps.backend.runtime.tools.gguf_converter_types import (
    ConversionConfig,
    ConversionProgress,
    GGUFVerificationError,
    QuantizationType,
)

logger = logging.getLogger("backend.runtime.tools.gguf_converter")


class GGUFConversionCancelled(Exception):
    """Raised when a conversion is cancelled via a cooperative cancel signal."""


def convert_safetensors_to_gguf(
    config: ConversionConfig,
    progress_callback: Optional[Callable[[ConversionProgress], None]] = None,
    *,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> str:
    """Convert a Safetensors file to GGUF format.
    
    Args:
        config: Conversion configuration
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to the output GGUF file
    """
    progress = ConversionProgress(status="loading_config")
    
    def update_progress():
        if progress_callback:
            progress_callback(progress)

    def check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            progress.status = "cancelled"
            progress.error = "cancelled"
            update_progress()
            raise GGUFConversionCancelled("cancelled")
    
    update_progress()
    check_cancel()
    
    # Load model config
    config_path = _safetensors_source.resolve_config_json_path(config.config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    logger.info("Loaded config: %s", model_config.get("_class_name") or model_config.get("model_type") or "unknown")
    check_cancel()
    
    comfy_layout = bool(getattr(config, "comfy_layout", True))

    profile = _profiles.resolve_profile(model_config, comfy_layout=comfy_layout)
    requested_type = _quantization.requested_ggml_type(config.quantization)
    dtype_rules = profile.quant_policy.compile(quant=config.quantization, user_rules=config.tensor_type_overrides)

    if profile.arch is GGUFArch.LLAMA:
        arch = str(model_config.get("model_type") or "llama")
    else:
        arch = profile.arch.value

    metadata_config = (
        profile.planner.normalize_metadata(model_config) if profile.planner is not None else dict(model_config)
    )

    key_mapping: dict[str, str] = {}
    if profile.key_mapping is not None:
        key_mapping = profile.key_mapping.build(model_config)
    
    # Load safetensors
    progress.status = "loading_weights"
    update_progress()
    check_cancel()
    
    logger.info("Loading safetensors: %s", config.safetensors_path)
    
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _safetensors_source.open_safetensors_source(config.safetensors_path) as sf:
        tensor_names = list(sf.keys())
        check_cancel()

        if profile.layout is GGUFKeyLayout.COMFY_CODEX and profile.planner is not None:
            plans, key_mapping = profile.planner.plan(tensor_names, sf, requested_type, dtype_rules)
        else:
            plans = _tensor_planner.plan_tensors(tensor_names, sf, key_mapping, requested_type, dtype_rules)

        progress.total_steps = len(plans)
        check_cancel()

        writer = GGUFWriter(path=str(output_path), arch=arch)
        _metadata.add_basic_metadata(
            writer,
            arch,
            metadata_config,
            config.quantization,
            config_path=config_path,
            safetensors_path=config.safetensors_path,
        )
        writer.add_bool("codex.converter.comfy_layout", comfy_layout)

        for plan in plans:
            raw_dtype = None if plan.ggml_type in {GGMLQuantizationType.F16, GGMLQuantizationType.F32} else plan.ggml_type
            writer.add_tensor_info(
                plan.gguf_name,
                tensor_shape=plan.stored_shape,
                tensor_dtype=plan.stored_dtype,
                tensor_nbytes=plan.stored_nbytes,
                raw_dtype=raw_dtype,
            )

        progress.status = "converting"
        update_progress()
        check_cancel()

        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()

            assert writer.fout is not None
            out = writer.fout[0]
            writer.write_padding(out, out.tell())

            # Stream-write tensors in the same order used for tensor-info offsets.
            chunk_rows = 1024
            for i, plan in enumerate(plans):
                check_cancel()
                progress.current_step = i + 1
                progress.current_tensor = plan.gguf_name
                update_progress()

                bytes_written = 0

                if plan.op == "copy":
                    sl = sf.get_slice(plan.src_name)
                    shape = tuple(int(x) for x in sl.get_shape())
                    if shape != plan.raw_shape:
                        raise RuntimeError(
                            f"Tensor shape changed during conversion for {plan.src_name}: {shape} vs {plan.raw_shape}"
                        )

                    if plan.ggml_type == GGMLQuantizationType.F16:
                        target_dtype = torch.float16
                        if len(shape) == 1:
                            t = sl[:].to(target_dtype).contiguous()
                            out.write(t.numpy().tobytes(order="C"))
                            bytes_written += t.numel() * 2
                        elif len(shape) == 2:
                            rows = shape[0]
                            for start in range(0, rows, chunk_rows):
                                check_cancel()
                                chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                                out.write(chunk.numpy().tobytes(order="C"))
                                bytes_written += chunk.numel() * 2
                        else:
                            t = sf.get_tensor(plan.src_name).to(target_dtype).contiguous()
                            out.write(t.numpy().tobytes(order="C"))
                            bytes_written += t.numel() * 2

                    elif plan.ggml_type == GGMLQuantizationType.F32:
                        target_dtype = torch.float32
                        if len(shape) == 1:
                            t = sl[:].to(target_dtype).contiguous()
                            out.write(t.numpy().tobytes(order="C"))
                            bytes_written += t.numel() * 4
                        elif len(shape) == 2:
                            rows = shape[0]
                            for start in range(0, rows, chunk_rows):
                                check_cancel()
                                chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                                out.write(chunk.numpy().tobytes(order="C"))
                                bytes_written += chunk.numel() * 4
                        else:
                            t = sf.get_tensor(plan.src_name).to(target_dtype).contiguous()
                            out.write(t.numpy().tobytes(order="C"))
                            bytes_written += t.numel() * 4

                    else:
                        if len(shape) == 1:
                            # By policy we keep 1D tensors in F16, so this would indicate a planning bug.
                            raise RuntimeError(f"Unexpected quantized 1D tensor plan for {plan.src_name}: {shape}")

                        if len(shape) == 2:
                            rows = shape[0]
                            for start in range(0, rows, chunk_rows):
                                check_cancel()
                                chunk = sl[start : min(rows, start + chunk_rows)].to(torch.float32).contiguous()
                                arr = chunk.numpy()
                                try:
                                    q = quantize_numpy(arr, plan.ggml_type)
                                except Exception as exc:
                                    raise RuntimeError(
                                        f"Failed to quantize tensor {plan.src_name} to {plan.ggml_type.name}: {exc}"
                                    ) from exc
                                out.write(q.tobytes(order="C"))
                                bytes_written += q.nbytes
                        else:
                            t = sf.get_tensor(plan.src_name).to(torch.float32).contiguous()
                            arr = t.numpy()
                            try:
                                q = quantize_numpy(arr, plan.ggml_type)
                            except Exception as exc:
                                raise RuntimeError(
                                    f"Failed to quantize tensor {plan.src_name} to {plan.ggml_type.name}: {exc}"
                                ) from exc
                            out.write(q.tobytes(order="C"))
                            bytes_written += q.nbytes

                elif plan.op == "concat_dim0":
                    check_cancel()
                    if not plan.src_names:
                        raise RuntimeError(f"concat_dim0 plan has no sources for {plan.gguf_name}")

                    slices = [sf.get_slice(name) for name in plan.src_names]
                    shapes = [tuple(int(x) for x in sl.get_shape()) for sl in slices]
                    base_shape = shapes[0]
                    rank = len(base_shape)
                    if any(len(s) != rank for s in shapes[1:]):
                        raise RuntimeError(f"concat_dim0 source rank mismatch for {plan.gguf_name}: {shapes}")
                    if rank == 2:
                        trailing = base_shape[1:]
                        if any(s[1:] != trailing for s in shapes[1:]):
                            raise RuntimeError(
                                f"concat_dim0 source trailing dims mismatch for {plan.gguf_name}: {shapes}"
                            )

                    if rank == 1:
                        expected_shape = (sum(int(s[0]) for s in shapes),)
                    elif rank == 2:
                        expected_shape = (sum(int(s[0]) for s in shapes), int(base_shape[1]))
                    else:
                        raise RuntimeError(
                            f"concat_dim0 expects 1D/2D tensors for {plan.gguf_name}, got {base_shape}"
                        )

                    if expected_shape != plan.raw_shape:
                        raise RuntimeError(
                            f"concat_dim0 planned shape mismatch for {plan.gguf_name}: expected {expected_shape}, planned {plan.raw_shape}"
                        )

                    if plan.ggml_type == GGMLQuantizationType.F16:
                        target_dtype = torch.float16
                        if rank == 1:
                            for sl in slices:
                                check_cancel()
                                t = sl[:].to(target_dtype).contiguous()
                                out.write(t.numpy().tobytes(order="C"))
                                bytes_written += t.numel() * 2
                        else:
                            for sl, shape in zip(slices, shapes, strict=True):
                                rows = int(shape[0])
                                for start in range(0, rows, chunk_rows):
                                    check_cancel()
                                    chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                                    out.write(chunk.numpy().tobytes(order="C"))
                                    bytes_written += chunk.numel() * 2

                    elif plan.ggml_type == GGMLQuantizationType.F32:
                        target_dtype = torch.float32
                        if rank == 1:
                            for sl in slices:
                                check_cancel()
                                t = sl[:].to(target_dtype).contiguous()
                                out.write(t.numpy().tobytes(order="C"))
                                bytes_written += t.numel() * 4
                        else:
                            for sl, shape in zip(slices, shapes, strict=True):
                                rows = int(shape[0])
                                for start in range(0, rows, chunk_rows):
                                    check_cancel()
                                    chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                                    out.write(chunk.numpy().tobytes(order="C"))
                                    bytes_written += chunk.numel() * 4

                    else:
                        if rank != 2:
                            raise RuntimeError(
                                f"Unexpected quantized concat_dim0 tensor plan for {plan.gguf_name}: {base_shape}"
                            )
                        for sl, shape in zip(slices, shapes, strict=True):
                            rows = int(shape[0])
                            for start in range(0, rows, chunk_rows):
                                check_cancel()
                                chunk = sl[start : min(rows, start + chunk_rows)].to(torch.float32).contiguous()
                                arr = chunk.numpy()
                                try:
                                    q = quantize_numpy(arr, plan.ggml_type)
                                except Exception as exc:
                                    raise RuntimeError(
                                        f"Failed to quantize tensor {plan.gguf_name} to {plan.ggml_type.name}: {exc}"
                                    ) from exc
                                out.write(q.tobytes(order="C"))
                                bytes_written += q.nbytes

                else:
                    raise RuntimeError(f"Unknown tensor op for {plan.gguf_name}: {plan.op!r}")

                if bytes_written != plan.stored_nbytes:
                    raise RuntimeError(
                        f"Byte count mismatch for {plan.gguf_name}: wrote {bytes_written}, expected {plan.stored_nbytes}"
                    )
                writer.write_padding(out, plan.stored_nbytes)
        finally:
            writer.close()

    logger.info("GGUF file written: %s", output_path)
    check_cancel()
    
    # Verification step: validate the generated file
    progress.status = "verifying"
    update_progress()
    check_cancel()
    
    _verify.verify_gguf_file(
        gguf_path=str(output_path),
        source_safetensors=config.safetensors_path,
        tensor_plans=plans,
        key_mapping=key_mapping,
    )
    
    progress.status = "complete"
    progress.current_step = progress.total_steps
    update_progress()
    
    logger.info("GGUF conversion and verification complete: %s", output_path)
    return str(output_path)


__all__ = [
    "ConversionConfig",
    "ConversionProgress", 
    "GGUFConversionCancelled",
    "QuantizationType",
    "GGUFVerificationError",
    "convert_safetensors_to_gguf",
]
