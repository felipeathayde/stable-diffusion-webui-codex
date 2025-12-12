"""Debug helpers for Z Image runtime.

All logging here is opt-in via environment flags to avoid flooding normal runs.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Sequence

import torch


_TRUE = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        # Convenience: a single switch enables most ZImage debug flags.
        if name != "CODEX_ZIMAGE_DEBUG" and name.startswith("CODEX_ZIMAGE_DEBUG_"):
            global_raw = os.getenv("CODEX_ZIMAGE_DEBUG")
            if global_raw is not None and str(global_raw).strip().lower() in _TRUE:
                return True
        return bool(default)
    return str(raw).strip().lower() in _TRUE


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def truncate_text(text: str, *, limit: int = 300) -> str:
    if not isinstance(text, str):
        text = str(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def summarize_ints(values: Sequence[int], *, window: int = 8) -> str:
    if window <= 0:
        return "<suppressed>"
    if len(values) <= window * 2:
        return ",".join(str(v) for v in values)
    head = ",".join(str(v) for v in values[:window])
    tail = ",".join(str(v) for v in values[-window:])
    return f"{head},...,{tail}"


def find_indices(values: Sequence[int], needle: int, *, limit: int = 16) -> list[int]:
    out: list[int] = []
    for i, v in enumerate(values):
        if int(v) == int(needle):
            out.append(i)
            if len(out) >= limit:
                break
    return out


def tensor_stats(logger: logging.Logger, label: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        logger.info("[zimage-debug] %s: <none>", label)
        return
    if not torch.is_tensor(tensor):
        logger.info("[zimage-debug] %s: <non-tensor %s>", label, type(tensor).__name__)
        return
    with torch.no_grad():
        data = tensor.detach()
        stats_tensor = data.float()
        logger.info(
            "[zimage-debug] %s: shape=%s dtype=%s device=%s min=%.6g max=%.6g mean=%.6g std=%.6g norm=%.6g",
            label,
            tuple(data.shape),
            data.dtype,
            data.device,
            float(stats_tensor.min().item()),
            float(stats_tensor.max().item()),
            float(stats_tensor.mean().item()),
            float(stats_tensor.std(unbiased=False).item()),
            float(stats_tensor.norm().item()),
        )


__all__ = [
    "env_flag",
    "env_int",
    "find_indices",
    "summarize_ints",
    "tensor_stats",
    "truncate_text",
]
