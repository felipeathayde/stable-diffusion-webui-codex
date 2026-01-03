"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GGUF → state_dict loader with optional dequantization.
Loads GGUF tensor blobs and metadata using `GGUFReader`, returning either raw float tensors or `CodexParameter` entries for deferred/on-demand dequantization.

Symbols (top-level; keep in sync; no ghosts):
- `load_gguf_state_dict` (function): Loads a GGUF file into a PyTorch-style state dict (optionally dequantizing tensors).
- `get_gguf_metadata` (function): Extracts GGUF metadata fields into a JSON-serializable dict.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

import torch

logger = logging.getLogger("backend.quantization.gguf_loader")


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
    from apps.backend.quantization.gguf import GGMLQuantizationType, GGUFReader
    from .api import dequantize as quant_dequantize
    from .tensor import CodexParameter
    
    logger.info("Loading GGUF file: %s", gguf_path)
    reader = GGUFReader(gguf_path)
    
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
            state_dict[name] = torch.nn.Parameter(torch.from_numpy(tensor.data), requires_grad=False)
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
    
    reader = GGUFReader(gguf_path)
    metadata = {}
    
    for field in reader.fields.values():
        # Convert field to Python type
        name = field.name
        value = field.parts[-1].tolist() if hasattr(field.parts[-1], 'tolist') else field.parts[-1]
        
        # Handle bytes
        if isinstance(value, bytes):
            try:
                value = value.decode('utf-8')
            except:
                pass
        
        metadata[name] = value
    
    return metadata


__all__ = [
    "load_gguf_state_dict",
    "get_gguf_metadata",
]
