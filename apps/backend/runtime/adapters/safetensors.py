from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

import safetensors
import safetensors.torch as sf
import torch


class SafeTensorError(RuntimeError):
    pass


@contextmanager
def open_safetensor(path: str) -> Iterator[sf.SafeTensor]:
    try:
        handle = sf.safe_open(path, framework="pt")
    except (safetensors.SafetensorError, FileNotFoundError) as exc:
        raise SafeTensorError(f"Unable to open safetensor '{path}': {exc}") from exc
    try:
        yield handle
    finally:
        handle.close()


def load_tensors(path: str) -> Mapping[str, torch.Tensor]:
    with open_safetensor(path) as handle:
        keys = list(handle.keys())
        tensors = {key: handle.get_tensor(key) for key in keys}
    return tensors
