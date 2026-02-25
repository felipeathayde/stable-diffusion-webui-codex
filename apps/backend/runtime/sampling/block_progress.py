"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Block-level progress callback contract for sampling runtimes.
Defines the canonical transformer-options key used to propagate per-block progress callbacks from the sampling driver into model block loops.

Symbols (top-level; keep in sync; no ghosts):
- `BlockProgressCallback` (type alias): Callable contract receiving `(block_index, total_blocks)` as 1-based progress.
- `BLOCK_PROGRESS_CALLBACK_KEY` (constant): Canonical transformer-options key for block progress callback injection.
- `resolve_block_progress_callback` (function): Validate and resolve optional block progress callback from transformer options.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable


BlockProgressCallback = Callable[[int, int], None]

BLOCK_PROGRESS_CALLBACK_KEY = "codex_sampling_block_progress_callback"


def resolve_block_progress_callback(
    transformer_options: Mapping[str, Any] | None,
) -> BlockProgressCallback | None:
    if transformer_options is None:
        return None
    if not isinstance(transformer_options, Mapping):
        raise RuntimeError(
            "transformer_options must be Mapping[str, Any] when resolving block progress callback "
            f"(got {type(transformer_options).__name__})."
        )
    raw_callback = transformer_options.get(BLOCK_PROGRESS_CALLBACK_KEY, None)
    if raw_callback is None:
        return None
    if not callable(raw_callback):
        raise RuntimeError(
            f"transformer_options['{BLOCK_PROGRESS_CALLBACK_KEY}'] must be callable when provided "
            f"(got {type(raw_callback).__name__})."
        )
    return raw_callback


__all__ = [
    "BlockProgressCallback",
    "BLOCK_PROGRESS_CALLBACK_KEY",
    "resolve_block_progress_callback",
]
