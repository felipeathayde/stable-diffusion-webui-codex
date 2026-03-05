"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime tools facade exposing heavyweight offline-style utilities (GGUF conversion, SafeTensors merge, and CodexPack packing).
Re-exports public runtime tool APIs used by `/api/tools/*` and CLI-like tooling.

Symbols (top-level; keep in sync; no ghosts):
- `ConversionConfig` (class): Conversion configuration for SafeTensors → GGUF (inputs, outputs, verification flags).
- `ConversionProgress` (class): Progress callback payload emitted by the converter.
- `QuantizationType` (class): Quantization enum/type used by the converter.
- `convert_safetensors_to_gguf` (function): Convert SafeTensors weights (including sharded indexes) to GGUF.
- `SafetensorsMergeConfig` (class): Merge configuration for collapsing safetensors sources into one file.
- `SafetensorsMergeProgress` (class): Progress callback payload emitted by the safetensors merge tool.
- `merge_safetensors_source` (function): Merge a safetensors source (file/index/dir) into one `.safetensors` file.
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
from .safetensors_merge import SafetensorsMergeConfig, SafetensorsMergeProgress, merge_safetensors_source
from .codexpack_packer import CodexPackPackError, pack_gguf_to_codexpack_v1

__all__ = [
    "CodexPackPackError",
    "ConversionConfig",
    "ConversionProgress",
    "QuantizationType",
    "SafetensorsMergeConfig",
    "SafetensorsMergeProgress",
    "convert_safetensors_to_gguf",
    "merge_safetensors_source",
    "pack_gguf_to_codexpack_v1",
]
