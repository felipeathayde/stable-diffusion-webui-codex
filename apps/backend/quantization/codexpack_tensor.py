"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime packed-weight containers for CodexPack GGUF execution.
Defines `torch.nn.Parameter` subclasses used by CodexPack loaders to attach kernel/layout metadata to packed tensors while preserving
GGUF invariants (byte-packed storage; `computation_dtype` controls output dtype, not storage dtype).

Symbols (top-level; keep in sync; no ghosts):
- `CodexPackLinearQ4KTilepackV1Parameter` (class): Packed Q4_K tilepack_v1 linear-weight container (I8 blob + metadata).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

__all__ = [
    "CodexPackLinearQ4KTilepackV1Parameter",
]


class CodexPackLinearQ4KTilepackV1Parameter(torch.nn.Parameter):
    """Packed Q4_K `tilepack_v1` linear weight (stored as an I8 blob; not a dequantized matrix)."""

    keymap_id: str
    kernel_id: str
    out_features: int
    in_features: int
    computation_dtype: torch.dtype
    dora_norm_out: Optional[torch.Tensor]

    def __new__(
        cls,
        data: Any,
        *,
        keymap_id: str,
        kernel_id: str,
        out_features: int,
        in_features: int,
        computation_dtype: torch.dtype = torch.float16,
        dora_norm_out: Optional[torch.Tensor] = None,
        requires_grad: bool = False,
    ):
        if not isinstance(keymap_id, str) or not keymap_id.strip():
            raise ValueError(f"keymap_id must be a non-empty string; got: {keymap_id!r}")
        if not isinstance(kernel_id, str) or not kernel_id.strip():
            raise ValueError(f"kernel_id must be a non-empty string; got: {kernel_id!r}")
        if int(out_features) <= 0 or int(in_features) <= 0:
            raise ValueError(f"out_features/in_features must be positive; got: {out_features!r}/{in_features!r}")

        if isinstance(data, np.ndarray):
            if not data.flags.writeable:
                data = np.array(data, copy=True)
            tensor = torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            tensor = data.detach()
        else:
            arr = np.asarray(data)
            if not arr.flags.writeable:
                arr = np.array(arr, copy=True)
            tensor = torch.from_numpy(arr)

        if tensor.dtype not in (torch.int8, torch.uint8):
            raise TypeError(
                "CodexPackLinearQ4KTilepackV1Parameter expects packed bytes with dtype int8/uint8; "
                f"got: {tensor.dtype}"
            )

        instance = super().__new__(cls, tensor, requires_grad=requires_grad)
        return instance

    def __init__(
        self,
        data: Any,
        *,
        keymap_id: str,
        kernel_id: str,
        out_features: int,
        in_features: int,
        computation_dtype: torch.dtype = torch.float16,
        dora_norm_out: Optional[torch.Tensor] = None,
        requires_grad: bool = False,
    ):
        super().__init__()
        self.keymap_id = str(keymap_id)
        self.kernel_id = str(kernel_id)
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.computation_dtype = computation_dtype
        self.dora_norm_out = dora_norm_out

    def copy_with_data(self, new_data: torch.Tensor) -> "CodexPackLinearQ4KTilepackV1Parameter":
        new = CodexPackLinearQ4KTilepackV1Parameter.__new__(
            CodexPackLinearQ4KTilepackV1Parameter,
            new_data,
            keymap_id=self.keymap_id,
            kernel_id=self.kernel_id,
            out_features=self.out_features,
            in_features=self.in_features,
            computation_dtype=self.computation_dtype,
            dora_norm_out=self.dora_norm_out,
            requires_grad=self.requires_grad,
        )
        new.keymap_id = self.keymap_id
        new.kernel_id = self.kernel_id
        new.out_features = self.out_features
        new.in_features = self.in_features
        new.computation_dtype = self.computation_dtype
        new.dora_norm_out = self.dora_norm_out
        return new

    def to(self, *args, **kwargs) -> "CodexPackLinearQ4KTilepackV1Parameter":
        # Packed tensors store bytes; never cast storage dtype.
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
            raise TypeError(
                f"{self.__class__.__name__}.to() expected at most 4 positional arguments, got {len(args)}"
            )

        move_kwargs: dict[str, object] = {"non_blocking": non_blocking}
        if device is not None:
            move_kwargs["device"] = device
        if copy:
            move_kwargs["copy"] = True
        if memory_format is not None:
            move_kwargs["memory_format"] = memory_format

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

        if self.dora_norm_out is not None:
            new.dora_norm_out = self.dora_norm_out.to(**move_kwargs)

        return new

    def pin_memory(self, device=None) -> "CodexPackLinearQ4KTilepackV1Parameter":
        new = self.copy_with_data(torch.Tensor.pin_memory(self, device=device))
        if self.dora_norm_out is not None:
            new.dora_norm_out = self.dora_norm_out.pin_memory(device=device)
        return new

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(keymap_id={self.keymap_id!r}, kernel_id={self.kernel_id!r}, "
            f"shape=[{self.out_features}, {self.in_features}], computation_dtype={self.computation_dtype})"
        )
