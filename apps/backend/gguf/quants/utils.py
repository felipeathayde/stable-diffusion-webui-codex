from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from numpy.typing import DTypeLike

from ..constants import GGML_QUANT_SIZES, GGMLQuantizationType


def quick_split(tensor: torch.Tensor, partition: Sequence[int]) -> tuple[torch.Tensor, ...]:
    """Split tensor columns according to partition; final chunk absorbs remainder."""
    return torch.split(tensor, (*partition, tensor.shape[1] - sum(partition)), dim=-1)


def quant_shape_to_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(
            f"Quantized tensor row size ({shape[-1]}) is not a multiple of {quant_type.name} block size ({block_size})"
        )
    return (*shape[:-1], shape[-1] // block_size * type_size)


def quant_shape_from_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % type_size != 0:
        raise ValueError(
            f"Quantized tensor bytes per row ({shape[-1]}) is not a multiple of {quant_type.name} type size ({type_size})"
        )
    return (*shape[:-1], shape[-1] // type_size * block_size)


def apply_over_grouped_rows(
    func: Callable[[np.ndarray], np.ndarray], arr: np.ndarray, otype: DTypeLike, oshape: tuple[int, ...]
) -> np.ndarray:
    rows = arr.reshape((-1, arr.shape[-1]))
    osize = 1
    for dim in oshape:
        osize *= dim
    out = np.empty(shape=osize, dtype=otype)
    n_groups = (rows.shape[0] // 16) or 1
    np.concatenate([func(group).ravel() for group in np.array_split(rows, n_groups)], axis=0, out=out)
    return out.reshape(oshape)


def np_roundf(values: np.ndarray) -> np.ndarray:
    abs_values = np.abs(values)
    floored = np.floor(abs_values)
    delta = floored + np.floor(2 * (abs_values - floored))
    return np.sign(values) * delta
