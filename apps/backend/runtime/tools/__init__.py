"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime tools facade exposing heavyweight offline-style utilities (e.g. GGUF conversion and CodexPack packing).
Re-exports the public GGUF converter API used by `/api/tools/*` and CLI-like tooling, plus the CodexPack packer entrypoint.

Symbols (top-level; keep in sync; no ghosts):
- `ConversionConfig` (class): Conversion configuration for SafeTensors → GGUF (inputs, outputs, verification flags).
- `ConversionProgress` (class): Progress callback payload emitted by the converter.
- `QuantizationType` (class): Quantization enum/type used by the converter.
- `convert_safetensors_to_gguf` (function): Convert SafeTensors weights (including sharded indexes) to GGUF.
- `CodexPackPackError` (class): Raised when CodexPack packing fails.
- `pack_gguf_to_codexpack_v1` (function): Convert a base GGUF into `*.codexpack.gguf` (packed Q4_K tilepack_v1).
- `__all__` (constant): Export list for the tools facade.
"""

from .gguf_converter import (
    ConversionConfig,
    ConversionProgress,
    QuantizationType,
    convert_safetensors_to_gguf,
)
from .codexpack_packer import CodexPackPackError, pack_gguf_to_codexpack_v1

__all__ = [
    "CodexPackPackError",
    "ConversionConfig",
    "ConversionProgress",
    "QuantizationType",
    "convert_safetensors_to_gguf",
    "pack_gguf_to_codexpack_v1",
]
