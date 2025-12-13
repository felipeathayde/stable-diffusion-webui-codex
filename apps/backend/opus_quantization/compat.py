# OpusQuantization - Compatibility bridge with old GGUF code

"""
This module provides compatibility with the existing ParameterGGUF system.

Usage:
    # In operations_gguf.py, replace:
    from apps.backend.gguf.quants.kernels.base import ParameterGGUF
    
    # With:
    from apps.backend.opus_quantization.compat import ParameterGGUF
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .core import QuantType, get_quant_spec, BLOCK_SIZES
from .tensor import OpusParameter
from .api import dequantize, bake

__all__ = [
    "ParameterGGUF",
    "dequantize_tensor", 
    "GGMLQuantizationType",
    "map_ggml_to_opus",
]


# Map from GGML enum values to OpusQuantization QuantType
# This bridges the old gguf.constants.GGMLQuantizationType to our QuantType
_GGML_TO_OPUS = {
    2: QuantType.Q4_0,
    3: QuantType.Q4_1,
    6: QuantType.Q5_0,
    7: QuantType.Q5_1,
    8: QuantType.Q8_0,
    10: QuantType.Q2_K,
    11: QuantType.Q3_K,
    12: QuantType.Q4_K,
    13: QuantType.Q5_K,
    14: QuantType.Q6_K,
    30: QuantType.BF16,
}


def map_ggml_to_opus(ggml_type: Any) -> Optional[QuantType]:
    """
    Map a GGML quantization type to OpusQuantization QuantType.
    
    Args:
        ggml_type: Either an int value or GGMLQuantizationType enum
        
    Returns:
        Corresponding QuantType, or None if not supported
    """
    if isinstance(ggml_type, int):
        return _GGML_TO_OPUS.get(ggml_type)
    
    # Handle GGMLQuantizationType enum
    if hasattr(ggml_type, 'value'):
        return _GGML_TO_OPUS.get(ggml_type.value)
    
    return None


class GGMLQuantizationType:
    """
    Compatibility shim for GGMLQuantizationType.
    
    Allows old code using GGMLQuantizationType.Q4_0 etc to work.
    """
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    BF16 = 30


class ParameterGGUF(torch.nn.Parameter):
    """
    Drop-in replacement for the old ParameterGGUF.
    
    This class maintains API compatibility while using OpusQuantization internally.
    """
    
    def __init__(self, tensor=None, requires_grad=False, no_init=False):
        super().__init__()
        if no_init:
            return
        
        # Map GGML type to Opus type
        ggml_type = getattr(tensor, 'tensor_type', None)
        self.qtype = map_ggml_to_opus(ggml_type)
        
        # Store shape in GGML order (reversed)
        if hasattr(tensor, 'shape'):
            self.real_shape = torch.Size(reversed(list(tensor.shape)))
        else:
            self.real_shape = torch.Size([])
        
        self.computation_dtype = torch.float16
        self.baked = False
        
        # Legacy compatibility - some code checks gguf_cls
        self.gguf_cls = _OpusQuantBridge(self.qtype) if self.qtype else None
    
    @property
    def shape(self):
        return self.real_shape
    
    def __new__(cls, tensor=None, requires_grad=False, no_init=False):
        # Handle numpy arrays
        if tensor is None:
            base = torch.empty(1)
        else:
            src = tensor.data if hasattr(tensor, 'data') else tensor
            if isinstance(src, torch.Tensor):
                base = src.detach().clone()
            else:
                try:
                    arr = np.asarray(src)
                    if not arr.flags.writeable:
                        arr = np.array(arr, copy=True)
                    base = torch.from_numpy(arr)
                except Exception:
                    base = torch.tensor(src)
        
        return super().__new__(cls, base, requires_grad=requires_grad)
    
    def dequantize_as_pytorch_parameter(self):
        """Dequantize and return as a regular Parameter."""
        if self.gguf_cls is not None:
            self.gguf_cls.bake(self)
        return torch.nn.Parameter(dequantize_tensor(self), requires_grad=False)
    
    def copy_with_data(self, data):
        """Create a copy with different underlying data but same metadata."""
        new = ParameterGGUF.__new__(ParameterGGUF, data, requires_grad=self.requires_grad, no_init=True)
        new.qtype = self.qtype
        new.real_shape = self.real_shape
        new.computation_dtype = self.computation_dtype
        new.baked = self.baked
        new.gguf_cls = self.gguf_cls
        return new
    
    def to(self, *args, **kwargs):
        """Move tensor and bake if needed.

        Important invariant for GGUF quantized tensors:
        - The underlying storage is byte-packed (typically uint8) and must NOT be
          cast to floating dtypes. Only `computation_dtype` controls the output
          dtype of dequantization.

        This allows callers like `nested_move_to_device(..., dtype=...)` to move
        GGUF tensors without corrupting their packed layout.
        """

        # Fast path: non-quantized tensors behave like a normal Parameter.
        if self.gguf_cls is None:
            return self.copy_with_data(self.data.to(*args, **kwargs))

        # Parse the common Tensor.to calling conventions (device/dtype/non_blocking/copy).
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        non_blocking = bool(kwargs.get("non_blocking", False))
        copy = bool(kwargs.get("copy", False))
        memory_format = kwargs.get("memory_format", None)

        if len(args) == 1:
            arg0 = args[0]
            if isinstance(arg0, torch.dtype):
                dtype = arg0
            elif isinstance(arg0, torch.Tensor):
                device = arg0.device
                dtype = arg0.dtype
            else:
                device = arg0
        elif len(args) == 2:
            arg0, arg1 = args
            if isinstance(arg0, torch.dtype):
                dtype = arg0
                non_blocking = bool(arg1)
            elif isinstance(arg0, torch.Tensor):
                device = arg0.device
                dtype = arg0.dtype
                non_blocking = bool(arg1)
            else:
                device = arg0
                dtype = arg1
        elif len(args) == 3:
            arg0, arg1, arg2 = args
            if isinstance(arg0, torch.dtype):
                dtype = arg0
                non_blocking = bool(arg1)
                copy = bool(arg2)
            elif isinstance(arg0, torch.Tensor):
                device = arg0.device
                dtype = arg0.dtype
                non_blocking = bool(arg1)
                copy = bool(arg2)
            else:
                device = arg0
                dtype = arg1
                non_blocking = bool(arg2)
        elif len(args) == 4:
            device, dtype, non_blocking, copy = args
            non_blocking = bool(non_blocking)
            copy = bool(copy)
        elif len(args) > 4:
            raise TypeError(f"ParameterGGUF.to() expected at most 4 positional arguments, got {len(args)}")

        # Move storage bytes without dtype casting. Keep memory_format only if provided.
        move_kwargs: dict[str, object] = {"non_blocking": non_blocking}
        if device is not None:
            move_kwargs["device"] = device
        if copy:
            move_kwargs["copy"] = True
        if memory_format is not None:
            move_kwargs["memory_format"] = memory_format

        moved = self.data.to(**move_kwargs)
        new = self.copy_with_data(moved)

        # Update dequantization compute dtype (but never cast packed storage).
        if isinstance(dtype, torch.dtype):
            new.computation_dtype = dtype
        
        # Always bake when moving
        if not new.baked and new.gguf_cls is not None:
            new.gguf_cls.bake(new)
        
        return new
    
    def pin_memory(self, device=None):
        return self.copy_with_data(torch.Tensor.pin_memory(self, device=device))


class _OpusQuantBridge:
    """
    Bridge class that provides gguf_cls-like interface using Opus kernels.
    
    This allows code that does `tensor.gguf_cls.bake(tensor)` to work.
    """
    
    def __init__(self, qtype: Optional[QuantType]):
        self.qtype = qtype
    
    def bake(self, tensor):
        """Bake the tensor using Opus kernels."""
        if tensor.baked:
            return
        
        if self.qtype is None:
            tensor.baked = True
            return
        
        spec = get_quant_spec(self.qtype)
        if spec and spec.bake:
            spec.bake(tensor)
        
        tensor.baked = True
    
    def dequantize_pytorch(self, tensor):
        """Dequantize using Opus kernels."""
        if self.qtype is None:
            return tensor
        
        spec = get_quant_spec(self.qtype)
        if spec is None or spec.dequantize is None:
            return tensor
        
        # Ensure baked
        if not tensor.baked:
            self.bake(tensor)
        
        result = spec.dequantize(tensor.data, tensor.computation_dtype)
        
        if tensor.real_shape:
            result = result.view(tensor.real_shape)
        
        return result


def dequantize_tensor(tensor):
    """
    Dequantize a GGUF tensor - compatible with both old and new code.
    
    This is the main entry point for dequantization, used by operations.py.
    """
    if tensor is None:
        return None
    
    # Non-quantized tensor
    if not hasattr(tensor, 'gguf_cls') and not hasattr(tensor, 'qtype'):
        return tensor
    
    # Get the bridge/gguf_cls
    gguf_cls = getattr(tensor, 'gguf_cls', None)
    
    if gguf_cls is None:
        return tensor
    
    # Lazy bake
    if hasattr(tensor, 'baked') and not tensor.baked:
        gguf_cls.bake(tensor)
    
    return gguf_cls.dequantize_pytorch(tensor)
