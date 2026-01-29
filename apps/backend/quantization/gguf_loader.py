"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GGUF → state_dict loader with optional dequantization and CodexPack auto-detection.
Loads GGUF tensor blobs and metadata using `GGUFReader`, returning either raw float tensors, `CodexParameter` entries for deferred/on-demand
dequantization, or CodexPack packed-weight containers when a `*.codexpack.gguf` file is detected.
CodexPack files must not contain raw quant tensors outside `__codexpack__.*`; the loader fails loud to prevent silent per-forward dequant fallback.

Symbols (top-level; keep in sync; no ghosts):
- `_numpy_to_frozen_parameter` (function): Converts a NumPy tensor blob into a read-only `nn.Parameter` (copies when non-writable).
- `load_gguf_state_dict` (function): Loads a GGUF file into a PyTorch-style state dict (optionally dequantizing tensors).
- `get_gguf_metadata` (function): Extracts GGUF metadata fields into a JSON-serializable dict.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

import numpy as np
import torch

logger = logging.getLogger("backend.quantization.gguf_loader")

def _numpy_to_frozen_parameter(data: np.ndarray) -> torch.nn.Parameter:
    # GGUFReader uses memmap-backed arrays which are often non-writable (read-only).
    # `torch.from_numpy` warns on non-writable arrays; copy to guarantee safe semantics.
    if isinstance(data, np.ndarray) and not data.flags.writeable:
        data = np.array(data, copy=True)
    return torch.nn.Parameter(torch.from_numpy(data), requires_grad=False)


def load_gguf_state_dict(
    gguf_path: str,
    dequantize: bool = False,
    *,
    computation_dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    """Load a GGUF file and return tensors as a state dict.
    
    Args:
        gguf_path: Path to the GGUF file.
        dequantize: If True, dequantize quantized tensors to float. If False, keep as CodexParameter.
        computation_dtype: Target dtype for dequantized tensors (used when dequantize=False as well).
    
    Returns:
        Dictionary mapping tensor names to PyTorch tensors.
    """
    from apps.backend.quantization.codexpack_keymaps import SUPPORTED_CODEXPACK_KEYMAP_IDS, is_supported_codexpack_keymap_id
    from apps.backend.quantization.codexpack_tensor import CodexPackLinearQ4KTilepackV1Parameter
    from apps.backend.quantization.gguf import GGMLQuantizationType, GGUFReader
    from apps.backend.quantization.gguf.codexpack import (
        CODEXPACK_SCHEMA_VERSION,
        KERNEL_ID_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1,
        is_codexpack_gguf,
        load_codexpack_manifest_v1,
    )
    from .api import dequantize as quant_dequantize
    from .tensor import CodexParameter
    
    logger.info("Loading GGUF file: %s", gguf_path)
    reader = GGUFReader(gguf_path)

    if is_codexpack_gguf(reader):
        if dequantize:
            raise RuntimeError(
                "CodexPack GGUF does not support load-time dequantization. "
                "Load the base GGUF file instead of `*.codexpack.gguf` when using `--gguf-exec=dequant_upfront`."
            )
        if computation_dtype != torch.float16:
            raise RuntimeError(
                "CodexPack GGUF v1 only supports fp16 runtime compute for packed linears. "
                f"got computation_dtype={computation_dtype}."
            )

        pack = load_codexpack_manifest_v1(reader)
        if pack.schema_version != CODEXPACK_SCHEMA_VERSION:
            raise RuntimeError(
                f"CodexPack schema_version mismatch: expected {CODEXPACK_SCHEMA_VERSION}, got {pack.schema_version}"
            )
        if pack.kernel_id != KERNEL_ID_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1:
            raise RuntimeError(
                "CodexPack v1 loader only supports kernel_id "
                f"{KERNEL_ID_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1!r}; got: {pack.kernel_id!r}"
            )
        if not is_supported_codexpack_keymap_id(pack.keymap_id):
            allowed = ", ".join(sorted(SUPPORTED_CODEXPACK_KEYMAP_IDS))
            raise RuntimeError(
                "CodexPack keymap_id is unknown to this build. "
                f"got: {pack.keymap_id!r}. allowed: {allowed}"
            )

        tensors_by_name = {t.name: t for t in reader.tensors}
        packed_param_keys: set[str] = set()
        packed_internal_names: set[str] = set()
        packed_weights: dict[str, torch.Tensor] = {}

        manifest = pack.manifest
        entries = manifest.get("entries", [])
        if not isinstance(entries, list):
            raise RuntimeError("CodexPack manifest.entries must be a list (post-validation invariant).")

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("kind") != "linear_weight":
                continue
            param_key = entry.get("param_key")
            if not isinstance(param_key, str) or not param_key.strip():
                raise RuntimeError("CodexPack manifest entry has invalid param_key (post-validation invariant).")
            if param_key in packed_param_keys:
                raise RuntimeError(f"CodexPack manifest contains duplicate param_key: {param_key!r}")

            shape = entry.get("shape")
            if not isinstance(shape, list) or len(shape) != 2:
                raise RuntimeError(f"CodexPack entry {param_key!r} has invalid shape (post-validation invariant).")
            out_features, in_features = int(shape[0]), int(shape[1])

            tensors = entry.get("tensors")
            if not isinstance(tensors, dict):
                raise RuntimeError(f"CodexPack entry {param_key!r} has invalid tensors (post-validation invariant).")
            packed_name = tensors.get("packed")
            if not isinstance(packed_name, str):
                raise RuntimeError(f"CodexPack entry {param_key!r} has invalid tensors.packed (post-validation invariant).")
            dora_norm_name = tensors.get("dora_norm_out")
            if not isinstance(dora_norm_name, str):
                raise RuntimeError(
                    f"CodexPack entry {param_key!r} has invalid tensors.dora_norm_out (post-validation invariant)."
                )

            packed_internal_names.add(packed_name)
            packed_internal_names.add(dora_norm_name)

            packed_tensor = tensors_by_name[packed_name]
            dora_tensor = tensors_by_name[dora_norm_name]

            # GGUFReader exposes memmap-backed arrays which may be non-writable; avoid `torch.from_numpy` warnings
            # by letting the CodexPack parameter class handle the packed blob and explicitly copying float tensors.
            packed_blob = packed_tensor.data.reshape(-1)
            dora_norm_out = torch.from_numpy(np.array(dora_tensor.data, copy=True)).reshape(-1)

            packed_weights[param_key] = CodexPackLinearQ4KTilepackV1Parameter(
                packed_blob,
                keymap_id=pack.keymap_id,
                kernel_id=pack.kernel_id,
                out_features=out_features,
                in_features=in_features,
                computation_dtype=computation_dtype,
                dora_norm_out=dora_norm_out,
            )
            packed_param_keys.add(param_key)

        state_dict: dict[str, torch.Tensor] = {}
        for tensor in reader.tensors:
            name = tensor.name

            # Never surface internal CodexPack payload tensors to the model state_dict.
            if name in packed_internal_names or name.startswith("__codexpack__."):
                continue

            # Reject ambiguous files: the payload owns param_key names; the GGUF tensor table must not.
            if name in packed_param_keys:
                raise RuntimeError(
                    "CodexPack GGUF is ambiguous: found a GGUF tensor with a packed param_key name. "
                    f"name={name!r}."
                )

            ggml_type = tensor.tensor_type
            real_shape = tuple(int(v) for v in reversed(tensor.shape.tolist()))

            if ggml_type in {
                GGMLQuantizationType.F16,
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F64,
                GGMLQuantizationType.I8,
                GGMLQuantizationType.I16,
                GGMLQuantizationType.I32,
                GGMLQuantizationType.I64,
            }:
                state_dict[name] = _numpy_to_frozen_parameter(tensor.data)
                continue

            # CodexPack is an optimized artifact; leaving raw quant tensors would silently reintroduce
            # per-forward dequantization. Fail loud so the pack generator can be fixed.
            raise RuntimeError(
                "CodexPack GGUF must not contain raw quantized tensors outside `__codexpack__.*`. "
                f"found {ggml_type.name} tensor {name!r} with logical shape {real_shape}."
            )

        state_dict.update(packed_weights)
        logger.info("Loaded %d tensors from CodexPack GGUF", len(state_dict))
        return state_dict
    
    state_dict = {}
    
    for tensor in reader.tensors:
        name = tensor.name
        
        logger.debug("Tensor: %s, shape=%s, type=%s", 
                    name, tensor.shape, tensor.tensor_type)

        ggml_type = tensor.tensor_type
        # ReaderTensor.shape stores GGUF dims order; the actual tensor is reshaped as reversed(dims).
        real_shape = tuple(int(v) for v in reversed(tensor.shape.tolist()))

        if ggml_type in {
            GGMLQuantizationType.F16,
            GGMLQuantizationType.F32,
            GGMLQuantizationType.F64,
            GGMLQuantizationType.I8,
            GGMLQuantizationType.I16,
            GGMLQuantizationType.I32,
            GGMLQuantizationType.I64,
        }:
            state_dict[name] = _numpy_to_frozen_parameter(tensor.data)
            continue

        param = CodexParameter(
            tensor.data,
            qtype=ggml_type,
            shape=real_shape,
            computation_dtype=computation_dtype,
        )
        if dequantize:
            state_dict[name] = quant_dequantize(param)
        else:
            state_dict[name] = param
    
    logger.info("Loaded %d tensors from GGUF", len(state_dict))
    return state_dict


def get_gguf_metadata(gguf_path: str) -> Dict[str, Any]:
    """Get metadata from a GGUF file.
    
    Args:
        gguf_path: Path to the GGUF file.
    
    Returns:
        Dictionary with metadata fields.
    """
    from apps.backend.quantization.gguf import GGUFReader
    from apps.backend.quantization.gguf.constants import GGUFValueType
    
    reader = GGUFReader(gguf_path)
    metadata = {}
    
    for field in reader.fields.values():
        name = field.name
        if not field.types or not field.data or field.data[0] < 0:
            continue

        vtype = field.types[0]

        if vtype == GGUFValueType.STRING:
            raw = field.parts[field.data[0]]
            b = bytes(raw)
            try:
                metadata[name] = b.decode("utf-8")
            except UnicodeDecodeError:
                metadata[name] = b
            continue

        if vtype == GGUFValueType.ARRAY:
            if len(field.types) < 2:
                raise RuntimeError(f"GGUF ARRAY field has no item type: {name!r}")
            item_type = field.types[1]
            items: list[Any] = []
            for idx in field.data:
                part = field.parts[idx]
                if item_type == GGUFValueType.STRING:
                    b = bytes(part)
                    try:
                        items.append(b.decode("utf-8"))
                    except UnicodeDecodeError:
                        items.append(b)
                    continue
                if not isinstance(part, np.ndarray) or part.size != 1:
                    items.append(part.tolist() if isinstance(part, np.ndarray) else part)
                    continue
                items.append(part.reshape(()).item())
            metadata[name] = items
            continue

        part = field.parts[field.data[0]]
        if isinstance(part, np.ndarray) and part.size == 1:
            metadata[name] = part.reshape(()).item()
        else:
            metadata[name] = part.tolist() if isinstance(part, np.ndarray) else part
    
    return metadata


__all__ = [
    "load_gguf_state_dict",
    "get_gguf_metadata",
]
