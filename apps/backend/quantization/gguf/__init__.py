"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public re-exports for GGUF IO helpers (reader/writer/constants).
Provides a small facade over the GGUF schema, IO primitives, and quantized-shape helpers for use by runtime loaders and tools.

Symbols (top-level; keep in sync; no ghosts):
- `GGML_QUANT_SIZES` (constant): Mapping `{GGMLQuantizationType: (block_size, type_size)}`.
- `GGMLQuantizationType` (enum): GGML/GGUF quantization type identifiers.
- `LlamaFileType` (enum): GGUF file type identifiers (compat metadata).
- `GGUFReader` (class): Memmap-based GGUF parser.
- `GGUFWriter` (class): GGUF v3 writer (tensor info + KV store).
- `ReaderTensor` (class): Tensor descriptor returned by `GGUFReader` for each tensor.
- `CODEXPACK_SCHEMA` (constant): Required `codex.pack.schema` value for CodexPack GGUF files.
- `CODEXPACK_SCHEMA_VERSION` (constant): Supported CodexPack schema version.
- `KERNEL_ID_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1` (constant): CodexPack v1 CUDA kernel identifier for Q4_K tile-packed linears.
- `CUDA_SM_MIN_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1` (constant): Minimum CUDA SM required by `tilepack_v1` (SM86 baseline).
- `TILE_M_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1` (constant): Output-channel tile size for `tilepack_v1`.
- `TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1` (constant): Input-feature tile size for `tilepack_v1` (equals GGML Q4_K block size).
- `CodexPackError` (exception): Raised when a CodexPack GGUF fails schema/manifest validation.
- `CodexPackManifestV1` (dataclass): Parsed v1 CodexPack manifest wrapper.
- `is_codexpack_gguf` (function): Returns True when a GGUF file declares CodexPack schema keys.
- `load_codexpack_manifest_v1` (function): Loads and validates a v1 CodexPack manifest from a GGUF file.
- `quant_shape_from_byte_shape` (function): Converts packed byte shapes → logical tensor shapes for a quant type.
- `quant_shape_to_byte_shape` (function): Converts logical tensor shapes → packed byte shapes for a quant type.
"""

from .codexpack import (
    CODEXPACK_SCHEMA,
    CODEXPACK_SCHEMA_VERSION,
    KERNEL_ID_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1,
    CUDA_SM_MIN_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1,
    TILE_M_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1,
    TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1,
    CodexPackError,
    CodexPackManifestV1,
    is_codexpack_gguf,
    load_codexpack_manifest_v1,
)
from .constants import GGML_QUANT_SIZES, GGMLQuantizationType, LlamaFileType
from .quant_shapes import quant_shape_from_byte_shape, quant_shape_to_byte_shape
from .reader import GGUFReader, ReaderTensor
from .writer import GGUFWriter

__all__ = [
    "CODEXPACK_SCHEMA",
    "CODEXPACK_SCHEMA_VERSION",
    "KERNEL_ID_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1",
    "CUDA_SM_MIN_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1",
    "TILE_M_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1",
    "TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1",
    "CodexPackError",
    "CodexPackManifestV1",
    "GGML_QUANT_SIZES",
    "GGMLQuantizationType",
    "GGUFReader",
    "GGUFWriter",
    "LlamaFileType",
    "ReaderTensor",
    "is_codexpack_gguf",
    "load_codexpack_manifest_v1",
    "quant_shape_from_byte_shape",
    "quant_shape_to_byte_shape",
]
