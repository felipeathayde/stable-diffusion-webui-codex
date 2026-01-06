"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Opt-in debug helpers for Z-Image runtime.
Implements env-flag parsing and lightweight tensor/text summaries to support targeted diagnostics without flooding normal runs.

Symbols (top-level; keep in sync; no ghosts):
- `env_flag` (function): Reads a boolean env flag (supports global `CODEX_ZIMAGE_DEBUG` enablement).
- `env_int` (function): Reads an integer env flag with a default fallback.
- `truncate_text` (function): Truncates text to a configurable max length for logs.
- `summarize_ints` (function): Summarizes long integer sequences using a head/tail window.
- `find_indices` (function): Finds a bounded number of indices matching a value in a sequence.
- `tensor_stats` (function): Logs min/max/mean/std/norm for a tensor (no-grad, float stats).
- `__all__` (constant): Explicit export list for debug helpers.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Sequence

import torch

from apps.backend.infra.config.env_flags import env_flag as _env_flag
from apps.backend.infra.config.env_flags import env_int as _env_int


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        # Convenience: a single switch enables most ZImage debug flags.
        if name != "CODEX_ZIMAGE_DEBUG" and name.startswith("CODEX_ZIMAGE_DEBUG_") and _env_flag(
            "CODEX_ZIMAGE_DEBUG",
            default=False,
        ):
            return True
        return bool(default)
    return _env_flag(name, default=default)


def env_int(name: str, default: int) -> int:
    return _env_int(name, default)


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
