from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from math import log2, ceil
from typing import Any, Sequence

import numpy as np
import torch

from ....constants import GGML_QUANT_SIZES, GGMLQuantizationType, QK_K
from ....lazy import LazyNumpyTensor
from ....quick_4bits_ops import change_4bits_order, quick_unpack_4bits, quick_unpack_4bits_u
from ...registry import CodexQuantKernelBase, QuantError, QuantKernel, register_kernel
from ...utils import (
    apply_over_grouped_rows,
    quant_shape_from_byte_shape,
    quant_shape_to_byte_shape,
    quick_split,
)
from .forge_bf16 import (
    dequantize_numpy as _bf16_dequantize_numpy,
    dequantize_torch as _bf16_dequantize_torch,
    quantize_numpy as _bf16_quantize_numpy,
    quantize_torch as _bf16_quantize_torch,
)
from .forge_q4_0 import (
    dequantize_numpy as _q40_dequantize_numpy,
    dequantize_torch as _q40_dequantize_torch,
    quantize_numpy as _q40_quantize_numpy,
    quantize_torch as _q40_quantize_torch,
)
from .forge_q4_1 import (
    dequantize_numpy as _q41_dequantize_numpy,
    dequantize_torch as _q41_dequantize_torch,
    quantize_numpy as _q41_quantize_numpy,
    quantize_torch as _q41_quantize_torch,
)
from .forge_q5_0 import (
    dequantize_numpy as _q50_dequantize_numpy,
    dequantize_torch as _q50_dequantize_torch,
    quantize_numpy as _q50_quantize_numpy,
    quantize_torch as _q50_quantize_torch,
)
from .forge_q5_1 import (
    dequantize_numpy as _q51_dequantize_numpy,
    dequantize_torch as _q51_dequantize_torch,
    quantize_numpy as _q51_quantize_numpy,
    quantize_torch as _q51_quantize_torch,
)
from .forge_q8_0 import (
    dequantize_numpy as _q80_dequantize_numpy,
    dequantize_torch as _q80_dequantize_torch,
    quantize_numpy as _q80_quantize_numpy,
    quantize_torch as _q80_quantize_torch,
)
from ..k_family.forge_q2_k import (
    dequantize_numpy as _q2k_dequantize_numpy,
    dequantize_torch as _q2k_dequantize_torch,
)
from ..k_family.forge_q3_k import (
    dequantize_numpy as _q3k_dequantize_numpy,
    dequantize_torch as _q3k_dequantize_torch,
)
from ..k_family.forge_q4_k import (
    bake_pytorch as _q4k_bake_pytorch,
    dequantize_numpy as _q4k_dequantize_numpy,
    dequantize_torch as _q4k_dequantize_torch,
    get_scale_min_numpy as _q4k_scale_min_numpy,
    get_scale_min_pytorch as _q4k_scale_min_pytorch,
    K_SCALE_SIZE as _Q4_K_SCALE_SIZE,
)
from ..k_family.forge_q5_k import (
    dequantize_numpy as _q5k_dequantize_numpy,
    dequantize_torch as _q5k_dequantize_torch,
)
from ..k_family.forge_q6_k import (
    dequantize_numpy as _q6k_dequantize_numpy,
    dequantize_torch as _q6k_dequantize_torch,
)
from ..iq_family.forge_iq1_m import dequantize_numpy as _iq1m_dequantize_numpy
from ..iq_family.forge_iq1_s import (
    DELTA as _IQ1_S_DELTA,
    GRID_HEX as _IQ1_S_GRID_HEX,
    GRID_MAP as _IQ1_S_GRID_MAP,
    GRID_SHAPE as _IQ1_S_GRID_SHAPE,
    dequantize_numpy as _iq1s_dequantize_numpy,
)
from ..iq_family.forge_iq2_s import (
    GRID_HEX as _IQ2_S_GRID_HEX,
    GRID_MAP as _IQ2_S_GRID_MAP,
    GRID_SHAPE as _IQ2_S_GRID_SHAPE,
    dequantize_numpy as _iq2s_dequantize_numpy,
)
from ..iq_family.forge_iq2_xs import (
    GRID_HEX as _IQ2_XS_GRID_HEX,
    GRID_MAP as _IQ2_XS_GRID_MAP,
    GRID_SHAPE as _IQ2_XS_GRID_SHAPE,
    dequantize_numpy as _iq2xs_dequantize_numpy,
)
from ..iq_family.forge_iq2_xxs import (
    GRID_HEX as _IQ2_XXS_GRID_HEX,
    GRID_MAP as _IQ2_XXS_GRID_MAP,
    GRID_SHAPE as _IQ2_XXS_GRID_SHAPE,
    KSIGNS as _IQ2_XXS_KSIGNS,
    dequantize_numpy as _iq2xxs_dequantize_numpy,
)
from ..iq_family.forge_iq3_s import (
    GRID_HEX as _IQ3_S_GRID_HEX,
    GRID_MAP as _IQ3_S_GRID_MAP,
    GRID_SHAPE as _IQ3_S_GRID_SHAPE,
    dequantize_numpy as _iq3s_dequantize_numpy,
)
from ..iq_family.forge_iq3_xxs import (
    GRID_HEX as _IQ3_XXS_GRID_HEX,
    GRID_MAP as _IQ3_XXS_GRID_MAP,
    GRID_SHAPE as _IQ3_XXS_GRID_SHAPE,
    dequantize_numpy as _iq3xxs_dequantize_numpy,
)
from ..iq_family.forge_iq4_nl import (
    KVALUES as _IQ4_NL_KVALUES,
    dequantize_numpy as _iq4nl_dequantize_numpy,
)
from ..iq_family.forge_iq4_xs import dequantize_numpy as _iq4xs_dequantize_numpy


logger = logging.getLogger(__name__)


class CodexQuantKernel(CodexQuantKernelBase, ABC):
    qtype: GGMLQuantizationType
    block_size: int
    type_size: int

    grid: np.ndarray[Any, np.dtype[np.float32]] | None = None
    grid_shape: tuple[int, int] = (0, 0)
    grid_map: tuple[int | float, ...] = ()
    grid_hex: bytes | None = None

    def __init__(self):
        raise TypeError("Quant conversion classes can't have instances")

    def __init_subclass__(cls, qtype: GGMLQuantizationType) -> None:
        cls.qtype = qtype
        cls.block_size, cls.type_size = GGML_QUANT_SIZES[qtype]
        cls.__quantize_lazy = LazyNumpyTensor._wrap_fn(
            cls.__quantize_array,
            meta_noop=(np.uint8, cls.__shape_to_bytes)
        )
        cls.__dequantize_lazy = LazyNumpyTensor._wrap_fn(
            cls.__dequantize_array,
            meta_noop=(np.float32, cls.__shape_from_bytes)
        )
        register_kernel(QuantKernel(cls))


    @classmethod
    def init_grid(cls):
        if cls.grid is not None or cls.grid_hex is None:
            return

        bits_per_elem = ceil(log2(len(cls.grid_map)))
        assert bits_per_elem != 0, cls.qtype.name
        elems_per_byte = 8 // bits_per_elem

        grid = np.frombuffer(cls.grid_hex, dtype=np.uint8)
        # decode hexadecimal chars from grid
        grid = grid.reshape((-1, 2))
        grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
        grid = grid[..., 0] | grid[..., 1]
        # unpack the grid values
        grid = grid.reshape((-1, 1)) >> np.array([i for i in range(0, 8, 8 // elems_per_byte)], dtype=np.uint8).reshape((1, elems_per_byte))
        grid = (grid & ((1 << bits_per_elem) - 1)).reshape((-1, 1))
        grid_map = np.array(cls.grid_map, dtype=np.float32).reshape((1, -1))
        grid = np.take_along_axis(grid_map, grid, axis=-1)
        cls.grid = grid.reshape((1, 1, *cls.grid_shape))

    @classmethod
    def quantize_pytorch(cls, data, parent) -> torch.Tensor:
        if not parent.baked:
            raise ValueError('GGUF Tensor is not baked!')

        block_size, type_size = GGML_QUANT_SIZES[cls.qtype]
        blocks = data.reshape(-1, block_size)
        parent.data = cls.quantize_blocks_pytorch(blocks, block_size, type_size, parent).contiguous()
        return parent

    @classmethod
    def bake(cls, parameter):
        if parameter.baked:
            return

        data = parameter.data
        cls.block_size, cls.type_size = GGML_QUANT_SIZES[cls.qtype]
        rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
        n_blocks = rows.numel() // cls.type_size
        blocks = rows.reshape((n_blocks, cls.type_size))
        parameter.data = blocks.contiguous()
        cls.bake_inner(parameter)
        parameter.baked = True
        return

    @classmethod
    def bake_inner(cls, parameter):
        pass

    @classmethod
    def dequantize_pytorch(cls, x):
        if not x.baked:
            raise ValueError('GGUF Tensor is not baked!')

        blocks = cls.dequantize_blocks_pytorch(x.data, cls.block_size, cls.type_size, x)
        return blocks.view(x.shape)

    @classmethod
    @abstractmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def quantize_blocks_pytorch(cls, blocks, block_size, type_size, parent) -> torch.Tensor:
        raise NotImplementedError('Low bit LoRA for this data type is not implemented yet. Please select "Automatic (fp16 LoRA)" in "Diffusion in Low Bits" (on the top line of this page) to use this LoRA.')

    @classmethod
    @abstractmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def quantize_rows(cls, rows: np.ndarray) -> np.ndarray:
        rows = rows.astype(np.float32, copy=False)
        shape = rows.shape
        n_blocks = rows.size // cls.block_size
        blocks = rows.reshape((n_blocks, cls.block_size))
        blocks = cls.quantize_blocks(blocks)
        assert blocks.dtype == np.uint8
        assert blocks.shape[-1] == cls.type_size
        return blocks.reshape(cls.__shape_to_bytes(shape))

    @classmethod
    def dequantize_rows(cls, rows: np.ndarray) -> np.ndarray:
        rows = rows.view(np.uint8)
        shape = rows.shape
        n_blocks = rows.size // cls.type_size
        blocks = rows.reshape((n_blocks, cls.type_size))
        blocks = cls.dequantize_blocks(blocks)
        assert blocks.dtype == np.float32
        assert blocks.shape[-1] == cls.block_size
        return blocks.reshape(cls.__shape_from_bytes(shape))

    @classmethod
    def __shape_to_bytes(cls, shape: Sequence[int]):
        return quant_shape_to_byte_shape(shape, cls.qtype)

    @classmethod
    def __shape_from_bytes(cls, shape: Sequence[int]):
        return quant_shape_from_byte_shape(shape, cls.qtype)

    @classmethod
    def __quantize_array(cls, array: np.ndarray) -> np.ndarray:
        return apply_over_grouped_rows(
            cls.quantize_rows,
            arr=array,
            otype=np.uint8,
            oshape=cls.__shape_to_bytes(array.shape),
        )

    @classmethod
    def __dequantize_array(cls, array: np.ndarray) -> np.ndarray:
        cls.init_grid()
        return apply_over_grouped_rows(
            cls.dequantize_rows,
            arr=array,
            otype=np.float32,
            oshape=cls.__shape_from_bytes(array.shape),
        )

    @classmethod
    def __quantize_lazy(cls, lazy_tensor: LazyNumpyTensor, /) -> Any:
        pass

    @classmethod
    def __dequantize_lazy(cls, lazy_tensor: LazyNumpyTensor, /) -> Any:
        pass

    @classmethod
    def can_quantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> bool:
        return tensor.shape[-1] % cls.block_size == 0

    @classmethod
    def quantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> np.ndarray:
        if not cls.can_quantize(tensor):
            raise QuantError(f"Can't quantize tensor with shape {tensor.shape} to {cls.qtype.name}")
        if not isinstance(tensor, (np.ndarray, LazyNumpyTensor)):
            raise TypeError(f"Unsupported tensor type {type(tensor)!r} for quantization")
        if isinstance(tensor, LazyNumpyTensor):
            logger.debug(
                "quantize lazy tensor",
                extra={"qtype": cls.qtype.name, "shape": tuple(tensor.shape)},
            )
            return cls.__quantize_lazy(tensor)
        logger.debug(
            "quantize numpy tensor",
            extra={"qtype": cls.qtype.name, "shape": tuple(tensor.shape), "dtype": str(tensor.dtype)},
        )
        return cls.__quantize_array(tensor)

    @classmethod
    def dequantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> np.ndarray:
        if not isinstance(tensor, (np.ndarray, LazyNumpyTensor)):
            raise TypeError(f"Unsupported tensor type {type(tensor)!r} for dequantization")
        if isinstance(tensor, LazyNumpyTensor):
            logger.debug(
                "dequantize lazy tensor",
                extra={"qtype": cls.qtype.name, "shape": tuple(tensor.shape)},
            )
            return cls.__dequantize_lazy(tensor)
        logger.debug(
            "dequantize numpy tensor",
            extra={"qtype": cls.qtype.name, "shape": tuple(tensor.shape), "dtype": str(tensor.dtype)},
        )
        return cls.__dequantize_array(tensor)


class BF16(CodexQuantKernel, qtype=GGMLQuantizationType.BF16):
    @classmethod
    # same as ggml_compute_fp32_to_bf16 in ggml-impl.h
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _bf16_quantize_numpy(blocks)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _bf16_dequantize_numpy(blocks)

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        return _bf16_dequantize_torch(blocks)

class Q4_0(CodexQuantKernel, qtype=GGMLQuantizationType.Q4_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q40_quantize_numpy(blocks)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q40_dequantize_numpy(blocks)

    @classmethod
    def bake_inner(cls, parameter):
        blocks = parameter.data
        d, x = quick_split(blocks, [2])
        d = d.view(torch.float16).to(parameter.computation_dtype).view(torch.uint8)
        x = change_4bits_order(x).view(torch.uint8)
        parameter.data = torch.cat([d, x], dim=-1).contiguous()
        return

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        return _q40_dequantize_torch(blocks, parameter)

    @classmethod
    def quantize_blocks_pytorch(cls, blocks, block_size, type_size, parent) -> torch.Tensor:
        return _q40_quantize_torch(blocks, parent)


class Q4_1(CodexQuantKernel, qtype=GGMLQuantizationType.Q4_1):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q41_quantize_numpy(blocks)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q41_dequantize_numpy(blocks)

    @classmethod
    def bake_inner(cls, parameter):
        blocks = parameter.data

        d, m, qs = quick_split(blocks, [2, 2])
        d = d.view(torch.float16).to(parameter.computation_dtype).view(torch.uint8)
        m = m.view(torch.float16).to(parameter.computation_dtype).view(torch.uint8)
        qs = change_4bits_order(qs).view(torch.uint8)

        parameter.data = torch.cat([d, m, qs], dim=-1).contiguous()

        return

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        return _q41_dequantize_torch(blocks, parameter)

    @classmethod
    def quantize_blocks_pytorch(cls, blocks, block_size, type_size, parent) -> torch.Tensor:
        return _q41_quantize_torch(blocks, parent)


class Q5_0(CodexQuantKernel, qtype=GGMLQuantizationType.Q5_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q50_quantize_numpy(blocks)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q50_dequantize_numpy(blocks)

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        return _q50_dequantize_torch(blocks, parameter)

    @classmethod
    def quantize_blocks_pytorch(cls, blocks, block_size, type_size, parent) -> torch.Tensor:
        return _q50_quantize_torch(blocks, parent)


class Q5_1(CodexQuantKernel, qtype=GGMLQuantizationType.Q5_1):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q51_quantize_numpy(blocks)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q51_dequantize_numpy(blocks)

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        return _q51_dequantize_torch(blocks, parameter)

    @classmethod
    def quantize_blocks_pytorch(cls, blocks, block_size, type_size, parent) -> torch.Tensor:
        return _q51_quantize_torch(blocks, parent)


class Q8_0(CodexQuantKernel, qtype=GGMLQuantizationType.Q8_0):
    @classmethod
    # Implementation of Q8_0 with bit-exact same results as reference implementation in ggml-quants.c
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q80_quantize_numpy(blocks)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q80_dequantize_numpy(blocks)

    @classmethod
    def bake_inner(cls, parameter):
        blocks = parameter.data
        d, x = quick_split(blocks, [2])
        x = x.view(torch.int8)
        d = d.view(torch.float16).to(parameter.computation_dtype).view(torch.int8)
        parameter.data = torch.cat([d, x], dim=-1).contiguous()
        return

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        return _q80_dequantize_torch(blocks, parameter)

    @classmethod
    def quantize_blocks_pytorch(cls, blocks, block_size, type_size, parent) -> torch.Tensor:
        return _q80_quantize_torch(blocks, parent)


class Q2_K(CodexQuantKernel, qtype=GGMLQuantizationType.Q2_K):
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q2k_dequantize_numpy(blocks)

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        return _q2k_dequantize_torch(blocks, parameter)


class Q3_K(CodexQuantKernel, qtype=GGMLQuantizationType.Q3_K):
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q3k_dequantize_numpy(blocks)

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        return _q3k_dequantize_torch(blocks, parameter)


class Q4_K(CodexQuantKernel, qtype=GGMLQuantizationType.Q4_K):
    K_SCALE_SIZE = _Q4_K_SCALE_SIZE

    @staticmethod
    def get_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _q4k_scale_min_numpy(scales)

    @staticmethod
    def get_scale_min_pytorch(scales):
        return _q4k_scale_min_pytorch(scales)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q4k_dequantize_numpy(blocks)

    @classmethod
    def bake_inner(cls, parameter):
        _q4k_bake_pytorch(parameter)

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        return _q4k_dequantize_torch(blocks, parameter)


class Q5_K(CodexQuantKernel, qtype=GGMLQuantizationType.Q5_K):
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q5k_dequantize_numpy(blocks)

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        return _q5k_dequantize_torch(blocks, parameter)


class Q6_K(CodexQuantKernel, qtype=GGMLQuantizationType.Q6_K):
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _q6k_dequantize_numpy(blocks)

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        return _q6k_dequantize_torch(blocks, parameter)


class IQ2_XXS(CodexQuantKernel, qtype=GGMLQuantizationType.IQ2_XXS):
    ksigns: bytes = _IQ2_XXS_KSIGNS
    grid_shape = _IQ2_XXS_GRID_SHAPE
    grid_map = _IQ2_XXS_GRID_MAP
    grid_hex = _IQ2_XXS_GRID_HEX

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        if cls.grid is None:
            raise RuntimeError("IQ2_XXS grid failed to initialize")
        return _iq2xxs_dequantize_numpy(blocks, cls.grid)


class IQ2_XS(CodexQuantKernel, qtype=GGMLQuantizationType.IQ2_XS):
    grid_shape = _IQ2_XS_GRID_SHAPE
    grid_map = _IQ2_XS_GRID_MAP
    grid_hex = _IQ2_XS_GRID_HEX

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        if cls.grid is None:
            raise RuntimeError('IQ2_XS grid failed to initialize')
        return _iq2xs_dequantize_numpy(blocks, cls.grid)


class IQ2_S(CodexQuantKernel, qtype=GGMLQuantizationType.IQ2_S):
    grid_shape = _IQ2_S_GRID_SHAPE
    grid_map = _IQ2_S_GRID_MAP
    grid_hex = _IQ2_S_GRID_HEX

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        if cls.grid is None:
            raise RuntimeError('IQ2_S grid failed to initialize')
        return _iq2s_dequantize_numpy(blocks, cls.grid)


class IQ3_XXS(CodexQuantKernel, qtype=GGMLQuantizationType.IQ3_XXS):
    grid_shape = _IQ3_XXS_GRID_SHAPE
    grid_map = _IQ3_XXS_GRID_MAP
    grid_hex = _IQ3_XXS_GRID_HEX

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        if cls.grid is None:
            raise RuntimeError('IQ3_XXS grid failed to initialize')
        return _iq3xxs_dequantize_numpy(blocks, cls.grid)


class IQ3_S(CodexQuantKernel, qtype=GGMLQuantizationType.IQ3_S):
    grid_shape = _IQ3_S_GRID_SHAPE
    grid_map = _IQ3_S_GRID_MAP
    grid_hex = _IQ3_S_GRID_HEX

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        if cls.grid is None:
            raise RuntimeError('IQ3_S grid failed to initialize')
        return _iq3s_dequantize_numpy(blocks, cls.grid)


class IQ1_S(CodexQuantKernel, qtype=GGMLQuantizationType.IQ1_S):
    grid_shape = _IQ1_S_GRID_SHAPE
    grid_map = _IQ1_S_GRID_MAP
    grid_hex = _IQ1_S_GRID_HEX
    delta = _IQ1_S_DELTA

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        if cls.grid is None:
            raise RuntimeError('IQ1_S grid failed to initialize')
        return _iq1s_dequantize_numpy(blocks, cls.grid)


class IQ1_M(CodexQuantKernel, qtype=GGMLQuantizationType.IQ1_M):
    grid_shape = IQ1_S.grid_shape
    grid_map = IQ1_S.grid_map
    grid_hex = IQ1_S.grid_hex
    delta = IQ1_S.delta

    # Okay *this* type is weird. It's the only one which stores the f16 scales in multiple parts.
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        cls.init_grid()
        if cls.grid is None:
            raise RuntimeError('IQ1_M grid failed to initialize')
        return _iq1m_dequantize_numpy(blocks, cls.grid)


class IQ4_NL(CodexQuantKernel, qtype=GGMLQuantizationType.IQ4_NL):
    kvalues = _IQ4_NL_KVALUES

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _iq4nl_dequantize_numpy(blocks)


class IQ4_XS(CodexQuantKernel, qtype=GGMLQuantizationType.IQ4_XS):
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _iq4xs_dequantize_numpy(blocks)
