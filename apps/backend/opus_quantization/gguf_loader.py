"""GGUF file loader for text encoders.

This module provides utilities for loading GGUF (GPT-Generated Unified Format)
files as PyTorch state dicts, with automatic dequantization.

The GGUF format is commonly used for quantized LLM models (Q4_K, Q5_K, Q8_0, etc).
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger("backend.opus_quantization.gguf_loader")


def load_gguf_state_dict(
    gguf_path: str,
    dequantize: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load a GGUF file and return tensors as a state dict.
    
    Args:
        gguf_path: Path to the GGUF file.
        dequantize: If True, dequantize to fp16. If False, keep as ParameterGGUF.
    
    Returns:
        Dictionary mapping tensor names to PyTorch tensors.
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        raise ImportError(
            "The 'gguf' package is required for GGUF loading. "
            "Install it with: pip install gguf"
        )
    
    from .compat import ParameterGGUF, dequantize_tensor
    
    logger.info("Loading GGUF file: %s", gguf_path)
    reader = GGUFReader(gguf_path)
    
    state_dict = {}
    
    for tensor in reader.tensors:
        name = tensor.name
        
        logger.debug("Tensor: %s, shape=%s, type=%s", 
                    name, tensor.shape, tensor.tensor_type)
        
        if dequantize:
            # Create ParameterGGUF and immediately dequantize
            param = ParameterGGUF(tensor)
            pt_tensor = dequantize_tensor(param)
        else:
            # Keep as ParameterGGUF for lazy dequantization
            pt_tensor = ParameterGGUF(tensor)
        
        state_dict[name] = pt_tensor
    
    logger.info("Loaded %d tensors from GGUF", len(state_dict))
    return state_dict


def get_gguf_metadata(gguf_path: str) -> Dict[str, Any]:
    """Get metadata from a GGUF file.
    
    Args:
        gguf_path: Path to the GGUF file.
    
    Returns:
        Dictionary with metadata fields.
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        raise ImportError("The 'gguf' package is required for GGUF loading.")
    
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
