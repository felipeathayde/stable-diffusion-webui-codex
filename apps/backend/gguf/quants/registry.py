from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..constants import GGMLQuantizationType
from ..lazy import LazyNumpyTensor


logger = logging.getLogger(__name__)


class QuantError(Exception):
    """Raised when quantization or dequantization invariants are violated."""


@dataclass(slots=True)
class QuantKernel:
    """Registry entry describing how to quantize/dequantize a GGUF tensor."""

    impl: type["CodexQuantKernelBase"]

    @property
    def qtype(self) -> GGMLQuantizationType:
        return self.impl.qtype

    def quantize(self, tensor: np.ndarray | LazyNumpyTensor) -> np.ndarray:
        return self.impl.quantize(tensor)

    def dequantize(self, tensor: np.ndarray | LazyNumpyTensor) -> np.ndarray:
        return self.impl.dequantize(tensor)

    def quantize_pytorch(self, data: torch.Tensor, parent: Any) -> Any:
        return self.impl.quantize_pytorch(data, parent)

    def dequantize_pytorch(self, parameter: Any) -> torch.Tensor:
        return self.impl.dequantize_pytorch(parameter)

    def bake(self, parameter: Any) -> None:
        self.impl.bake(parameter)


_kernel_registry: dict[GGMLQuantizationType, QuantKernel] = {}


def register_kernel(kernel: QuantKernel) -> None:
    if not issubclass(kernel.impl, CodexQuantKernelBase):
        raise TypeError(f"Quant kernel implementation {kernel.impl.__name__} must inherit CodexQuantKernelBase")
    if kernel.qtype in _kernel_registry:
        raise ValueError(f"Quant kernel for {kernel.qtype.name} already registered")
    _kernel_registry[kernel.qtype] = kernel
    logger.debug("registered GGUF quant kernel", extra={"qtype": kernel.qtype.name, "impl": kernel.impl.__name__})


def get_kernel(qtype: GGMLQuantizationType) -> QuantKernel:
    if qtype not in _kernel_registry:
        raise NotImplementedError(f"Quantization for {qtype.name} is not yet implemented")
    return _kernel_registry[qtype]


def quantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise TypeError(f"quantize expects numpy.ndarray input, received {type(data)!r}")
    if qtype == GGMLQuantizationType.F32:
        return data.astype(np.float32, copy=False)
    if qtype == GGMLQuantizationType.F16:
        return data.astype(np.float16, copy=False)
    kernel = get_kernel(qtype)
    return kernel.quantize(data)


def dequantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise TypeError(f"dequantize expects numpy.ndarray input, received {type(data)!r}")
    if qtype == GGMLQuantizationType.F32:
        return data.view(np.float32)
    if qtype == GGMLQuantizationType.F16:
        return data.view(np.float16).astype(np.float32)
    kernel = get_kernel(qtype)
    return kernel.dequantize(data)


class CodexQuantKernelBase:
    """Forward-declared protocol; concrete implementation lives in kernels."""

    qtype: GGMLQuantizationType

    @classmethod
    def quantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def dequantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def quantize_pytorch(cls, data: torch.Tensor, parent: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def dequantize_pytorch(cls, parameter: Any) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def bake(cls, parameter: Any) -> None:
        raise NotImplementedError
