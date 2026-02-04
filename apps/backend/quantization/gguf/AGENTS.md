<!-- tags: backend, codex, gguf, io -->
# apps/backend/quantization/gguf Overview
Date: 2025-12-15
Last Review: 2026-01-29
Status: Active

## Purpose
- GGUF format IO and schema helpers (reader/writer/constants/quant-shape helpers) used by:
  - WebUI tools that convert checkpoints to GGUF.
  - Runtime loaders that need to parse GGUF tensor blobs and metadata.

## Key Files
- `constants.py` — GGUF file constants, enums, key namespace (`Keys`, `GGUFValueType`, `GGMLQuantizationType`, etc.).
- `codexpack.py` — CodexPack GGUF contract helpers (schema keys + strict manifest parsing/validation).
- `quant_shapes.py` — Quantized tensor shape conversion helpers.
- `reader.py` — `GGUFReader` (memmap-based GGUF parsing).
- `writer.py` — `GGUFWriter` (GGUF v3 writer, tensor info + KV store).

## Notes
- Keep this subpackage dependency-light: NumPy-only + stdlib.
- Quantization math lives in `apps/backend/quantization/kernels/*` (not here).
- 2026-01-02: Added standardized file header docstrings to GGUF helper modules (doc-only change; part of rollout).
- 2026-01-20: Removed unreferenced legacy helpers (`lazy.py`, `metadata.py`, `tensor_mapping.py`, `utility.py`, `vocab.py`); converter metadata injection lives in `apps/backend/runtime/tools/gguf_converter_metadata.py`.
- 2026-01-29: CodexPack manifest validation now recognizes `fallback_fp16_keys` (list of keys dequantized offline to FP16 when Q4_K tile alignment is not satisfied).
