"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public API surface for GGUF quantization in Codex.
Imports kernels for registration side-effects and re-exports core types (`QuantType`, `CodexParameter`) plus the high-level API (`dequantize`, `bake`, `quantize`).

Symbols (top-level; keep in sync; no ghosts):
- `QuantType` (enum): GGML/GGUF quantization type identifiers (alias of `GGMLQuantizationType`).
- `BLOCK_SIZES` (constant): Mapping `{QuantType: (block_size, type_size)}` used by kernels and IO.
- `CodexParameter` (class): Packed GGUF tensor wrapper used to hold quantized weights.
- `dequantize` (function): Dequantizes a `CodexParameter` into a floating-point `torch.Tensor`.
- `bake` (function): Prepares quantization-specific state (when required by a kernel).
- `quantize` (function): Quantizes a floating-point tensor into a `CodexParameter`/GGUF-compatible representation.
- `register_quant` (function): Registers quant types and their kernels into the quant registry.
- `get_quant_spec` (function): Retrieves the registered `QuantSpec` for a quant type.
- `__version__` (constant): Package version string for the quantization module.
"""

# Import kernels first to register them
from . import kernels  # noqa: F401 - triggers registration

from .core import QuantType, BLOCK_SIZES, register_quant, get_quant_spec
from .tensor import CodexParameter
from .api import dequantize, bake, quantize

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

__version__ = "2.0.0"  # Major version bump: ComfyUI-GGUF port
