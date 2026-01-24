"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Global GGUF execution mode selection for runtime codepaths.
Centralizes the meaning and parsing of the `--gguf-exec` flag used by checkpoint loaders and (future) packed-kernel execution.

Symbols (top-level; keep in sync; no ghosts):
- `GgufExecMode` (enum): Supported GGUF execution modes (`dequant_forward`, `dequant_upfront`, `cuda_pack`).
- `DEFAULT_GGUF_EXEC_MODE` (constant): Default exec mode when unset.
- `parse_gguf_exec_mode` (function): Parses a string into `GgufExecMode` (strict; raises on invalid).
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

from enum import Enum


class GgufExecMode(Enum):
    """How GGUF weights are handled at runtime."""

    DEQUANT_FORWARD = "dequant_forward"
    DEQUANT_UPFRONT = "dequant_upfront"
    CUDA_PACK = "cuda_pack"


DEFAULT_GGUF_EXEC_MODE = GgufExecMode.DEQUANT_FORWARD


def parse_gguf_exec_mode(raw: str) -> GgufExecMode:
    value = str(raw).strip().lower()
    for mode in GgufExecMode:
        if mode.value == value:
            return mode
    allowed = ", ".join(m.value for m in GgufExecMode)
    raise ValueError(f"--gguf-exec must be one of: {allowed}; got: {raw!r}")


__all__ = [
    "DEFAULT_GGUF_EXEC_MODE",
    "GgufExecMode",
    "parse_gguf_exec_mode",
]

