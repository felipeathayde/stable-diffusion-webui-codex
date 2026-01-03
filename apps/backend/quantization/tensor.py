# CodexQuantization - GGUF Tensor Parameter

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GGUF packed tensor parameter wrapper (`CodexParameter`).
Implements the canonical `torch.nn.Parameter` subclass used for GGUF/GGML quantized tensors, tracking quantization type, logical shape,
target computation dtype, and bake state.

Symbols (top-level; keep in sync; no ghosts):
- `CodexParameter` (class): Packed-weights parameter wrapper for GGUF tensors (metadata + device/dtype moves + bake hooks).
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .core import QuantType

__all__ = ["CodexParameter"]


class CodexParameter(torch.nn.Parameter):
    """
    PyTorch Parameter wrapper for GGUF quantized tensors.
    
    This is the canonical packed-parameter type used by the runtime for GGUF tensors
    with cleaner implementation and better separation of concerns.
    
    Attributes:
        qtype: The quantization type (QuantType enum)
        real_shape: The logical shape of the dequantized tensor
        computation_dtype: Target dtype for dequantized values
        baked: Whether the tensor has been pre-processed for fast dequant
    """
    
    # These are set after construction
    qtype: Optional["QuantType"]
    real_shape: torch.Size
    computation_dtype: torch.dtype
    baked: bool
    
    def __new__(
        cls,
        data: Any = None,
        requires_grad: bool = False,
        *,
        qtype: Optional["QuantType"] = None,
        shape: Optional[tuple] = None,
        computation_dtype: torch.dtype = torch.float16,
    ):
        """
        Create a new CodexParameter from data.
        
        Args:
            data: Raw quantized data (numpy array or torch tensor)
            requires_grad: Whether gradients are needed (usually False for GGUF)
            qtype: Quantization type
            shape: Logical shape of the dequantized tensor
            computation_dtype: Target dtype for computation
        """
        # Handle numpy arrays
        if isinstance(data, np.ndarray):
            if not data.flags.writeable:
                data = np.array(data, copy=True)
            tensor = torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            # Avoid eager clones here: `to()` already materializes a new tensor when needed,
            # and cloning on every call creates significant overhead for packed GGUF weights.
            tensor = data.detach()
        else:
            # Try to convert via numpy
            try:
                arr = np.asarray(data)
                if not arr.flags.writeable:
                    arr = np.array(arr, copy=True)
                tensor = torch.from_numpy(arr)
            except Exception:
                tensor = torch.tensor(data)
        
        instance = super().__new__(cls, tensor, requires_grad=requires_grad)
        return instance
    
    def __init__(
        self,
        data: Any = None,
        requires_grad: bool = False,
        *,
        qtype: Optional["QuantType"] = None,
        shape: Optional[tuple] = None,
        computation_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.qtype = qtype
        self.real_shape = torch.Size(shape) if shape else torch.Size([])
        self.computation_dtype = computation_dtype
        self.baked = False
    
    @property
    def shape(self) -> torch.Size:
        """Return the logical shape (dequantized), not the raw data shape."""
        if self.real_shape:
            return self.real_shape
        return self.data.shape
    
    def copy_with_data(self, new_data: torch.Tensor) -> "CodexParameter":
        """Create a copy with different underlying data but same metadata."""
        new = CodexParameter.__new__(
            CodexParameter,
            new_data,
            requires_grad=self.requires_grad,
            qtype=self.qtype,
            shape=tuple(self.real_shape) if self.real_shape else None,
            computation_dtype=self.computation_dtype,
        )
        new.qtype = self.qtype
        new.real_shape = self.real_shape
        new.computation_dtype = self.computation_dtype
        new.baked = self.baked
        return new
    
    def to(self, *args, **kwargs) -> "CodexParameter":
        """
        Move tensor to device/dtype.
        
        Automatically bakes the tensor if not already baked.
        """
        # Quantized tensors store packed bytes; never cast storage dtype.
        if self.qtype is not None:
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
                raise TypeError(f"CodexParameter.to() expected at most 4 positional arguments, got {len(args)}")

            move_kwargs: dict[str, object] = {"non_blocking": non_blocking}
            if device is not None:
                move_kwargs["device"] = device
            if copy:
                move_kwargs["copy"] = True
            if memory_format is not None:
                move_kwargs["memory_format"] = memory_format

            # Fast path: no-op move (avoid allocating a new CodexParameter wrapper and
            # cloning packed bytes) — only adjust computation_dtype when requested.
            if (
                not copy
                and memory_format is None
                and (device is None or torch.device(device) == self.data.device)
                and (len(move_kwargs) == 1 or (len(move_kwargs) == 2 and "device" in move_kwargs))
            ):
                if isinstance(dtype, torch.dtype):
                    self.computation_dtype = dtype
                return self

            new = self.copy_with_data(self.data.to(**move_kwargs))
            if isinstance(dtype, torch.dtype):
                new.computation_dtype = dtype
        else:
            new = self.copy_with_data(self.data.to(*args, **kwargs))
        
        # Always bake when moving - needed for dequantization
        if not new.baked and new.qtype is not None:
            from .api import bake
            bake(new)
        
        return new
    
    def pin_memory(self, device=None) -> "CodexParameter":
        """Pin memory for faster CPU->GPU transfer."""
        return self.copy_with_data(torch.Tensor.pin_memory(self, device=device))
    
    def dequantize(self) -> torch.Tensor:
        """
        Dequantize to a standard float tensor.
        
        Returns:
            Tensor of shape self.real_shape with dtype self.computation_dtype
        """
        from .api import dequantize
        return dequantize(self)
    
    def __repr__(self) -> str:
        qtype_name = self.qtype.name if self.qtype else "None"
        return (
            f"CodexParameter(qtype={qtype_name}, "
            f"shape={list(self.real_shape)}, "
            f"dtype={self.computation_dtype}, "
            f"baked={self.baked})"
        )
