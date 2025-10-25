from __future__ import annotations

import gc
import torch


def default_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def cpu() -> torch.device:
    return torch.device("cpu")


def torch_gc() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


__all__ = ["default_device", "cpu", "torch_gc"]

