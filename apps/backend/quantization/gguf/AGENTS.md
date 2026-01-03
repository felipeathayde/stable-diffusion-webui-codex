<!-- tags: backend, codex, gguf, io -->
# apps/backend/quantization/gguf Overview
Date: 2025-12-15
Owner: Runtime Maintainers
Last Review: 2026-01-02
Status: Active

## Purpose
- GGUF format IO and schema helpers (reader/writer/constants/metadata utilities) used by:
  - WebUI tools that convert checkpoints to GGUF.
  - Runtime loaders that need to parse GGUF tensor blobs and metadata.

## Key Files
- `constants.py` — GGUF file constants, enums, key namespace (`Keys`, `GGUFValueType`, `GGMLQuantizationType`, etc.).
- `reader.py` — `GGUFReader` (memmap-based GGUF parsing).
- `writer.py` — `GGUFWriter` (GGUF v3 writer, tensor info + KV store).

## Notes
- Keep this subpackage dependency-light: NumPy-only + stdlib.
- Quantization math lives in `apps/backend/quantization/kernels/*` (not here).
- 2026-01-02: Added standardized file header docstrings to GGUF helper modules (doc-only change; part of rollout).
