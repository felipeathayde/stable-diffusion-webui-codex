# OpusQuantization - Public API functions

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

import torch

from .core import get_quant_spec, QuantType, QUANT_REGISTRY

if TYPE_CHECKING:
    from .tensor import OpusParameter

logger = logging.getLogger(__name__)

__all__ = ["dequantize", "bake", "quantize"]


def bake(tensor: "OpusParameter") -> None:
    """
    Bake a GGUF tensor - pre-process for faster dequantization.
    
    This converts internal format to be optimal for repeated dequantization.
    Called automatically when moving tensor to a device.
    
    Args:
        tensor: OpusParameter to bake
    """
    if tensor.baked:
        return
    
    if tensor.qtype is None:
        tensor.baked = True
        return
    
    spec = get_quant_spec(tensor.qtype)
    if spec is None:
        logger.warning("No spec found for qtype %s, skipping bake", tensor.qtype)
        tensor.baked = True
        return
    
    if spec.bake is not None:
        try:
            spec.bake(tensor)
        except Exception as e:
            logger.error("Failed to bake tensor (qtype=%s): %s", tensor.qtype.name, e)
            raise
    
    tensor.baked = True


def dequantize(tensor: "OpusParameter") -> torch.Tensor:
    """
    Dequantize a GGUF tensor to float.
    
    Automatically bakes the tensor if not already baked.
    
    Args:
        tensor: OpusParameter to dequantize
        
    Returns:
        Dequantized tensor with shape tensor.real_shape and dtype tensor.computation_dtype
    """
    if tensor is None:
        return None
    
    # Handle non-quantized tensors
    if not hasattr(tensor, 'qtype') or tensor.qtype is None:
        return tensor
    
    # Lazy bake if needed
    if not tensor.baked:
        bake(tensor)
    
    spec = get_quant_spec(tensor.qtype)
    if spec is None:
        raise ValueError(f"Unknown quantization type: {tensor.qtype}")
    
    if spec.dequantize is None:
        raise ValueError(f"No dequantize kernel for type: {tensor.qtype.name}")
    
    # Dequantize
    result = spec.dequantize(tensor.data, tensor.computation_dtype)
    
    # Reshape to logical shape
    if tensor.real_shape:
        result = result.view(tensor.real_shape)
    
    return result


def quantize(
    tensor: torch.Tensor,
    qtype: QuantType,
    computation_dtype: torch.dtype = torch.float16,
) -> "OpusParameter":
    """
    Quantize a float tensor to GGUF format.
    
    Args:
        tensor: Float tensor to quantize
        qtype: Target quantization type
        computation_dtype: Dtype for dequantized computation
        
    Returns:
        OpusParameter with quantized data
        
    Raises:
        NotImplementedError: If quantization not implemented for this type
    """
    from .tensor import OpusParameter
    
    spec = get_quant_spec(qtype)
    if spec is None:
        raise ValueError(f"Unknown quantization type: {qtype}")
    
    if spec.quantize is None:
        raise NotImplementedError(
            f"Quantization not implemented for {qtype.name}. "
            f"This type only supports dequantization."
        )
    
    # Flatten and reshape for block processing
    original_shape = tensor.shape
    flat = tensor.flatten()
    
    # Ensure divisible by block size
    if flat.numel() % spec.block_size != 0:
        raise ValueError(
            f"Tensor size {flat.numel()} not divisible by block size {spec.block_size}"
        )
    
    n_blocks = flat.numel() // spec.block_size
    blocks = flat.view(n_blocks, spec.block_size)
    
    # Create temporary parameter for quantization context
    temp = OpusParameter.__new__(
        OpusParameter,
        torch.empty(1),
        qtype=qtype,
        shape=original_shape,
        computation_dtype=computation_dtype,
    )
    temp.computation_dtype = computation_dtype
    
    # Quantize
    quantized = spec.quantize(blocks, temp)
    
    # Create result parameter
    result = OpusParameter(
        quantized,
        qtype=qtype,
        shape=original_shape,
        computation_dtype=computation_dtype,
    )
    result.baked = False
    
    return result


# Convenience function for compatibility with old code
def dequantize_tensor(tensor) -> torch.Tensor:
    """
    Compatibility wrapper for old dequantize_tensor calls.
    
    Handles both OpusParameter and the old ParameterGGUF.
    """
    if tensor is None:
        return None
    
    # If it's already a regular tensor, return as-is
    if not hasattr(tensor, 'qtype') and not hasattr(tensor, 'gguf_cls'):
        return tensor
    
    # If it's an OpusParameter, use our dequantize
    if hasattr(tensor, 'qtype'):
        return dequantize(tensor)
    
    # If it's the old ParameterGGUF (has gguf_cls), use old method
    if hasattr(tensor, 'gguf_cls'):
        gguf_cls = tensor.gguf_cls
        if gguf_cls is None:
            return tensor
        
        # Lazy bake
        if hasattr(tensor, 'baked') and not tensor.baked:
            gguf_cls.bake(tensor)
        
        return gguf_cls.dequantize_pytorch(tensor)
    
    return tensor
