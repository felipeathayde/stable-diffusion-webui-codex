"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Import-light public facade for GGUF quantization in Codex.
Keeps `apps.backend.quantization.gguf` import-light (torch-free) while lazily re-exporting the torch-bound quantization API
(`dequantize`, `bake`, `quantize`) and core registry types (`QuantType`, `CodexParameter`) for convenience.

Symbols (top-level; keep in sync; no ghosts):
- `__getattr__` (function): Lazy import hook for torch-bound symbols.
- `__all__` (constant): Public module exports (lazy-resolved names).
- `__version__` (constant): Package version string for the quantization module.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    # Types
    "QuantType",
    "BLOCK_SIZES",
    "CodexParameter",
    # Functions
    "dequantize",
    "bake",
    "quantize",
    # Registry
    "register_quant",
    "get_quant_spec",
]

__version__ = "2.0.0"  # Major version bump: GGUF dequant port


def __getattr__(name: str) -> Any:  # pragma: no cover - import-time dispatch
    # Core registry/types (torch-bound due to type aliases in core.py).
    if name in {"QuantType", "BLOCK_SIZES", "register_quant", "get_quant_spec"}:
        _core = importlib.import_module(f"{__name__}.core")
        value = getattr(_core, name)
        globals()[name] = value
        return value

    # Tensor wrapper (torch-bound).
    if name == "CodexParameter":
        _tensor = importlib.import_module(f"{__name__}.tensor")
        value = _tensor.CodexParameter
        globals()[name] = value
        return value

    # High-level API (torch-bound; imports kernels for registration side-effects).
    if name in {"dequantize", "bake", "quantize"}:
        _api = importlib.import_module(f"{__name__}.api")
        value = getattr(_api, name)
        globals()[name] = value
        return value

    raise AttributeError(name)
