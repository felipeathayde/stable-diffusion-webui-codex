# apps/backend/gguf Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Houses utilities for working with GGUF-formatted checkpoints (metadata parsing, tensor mapping, quantization helpers) used by WAN22 and other GGUF-enabled engines.

## Key Files
- `gguf_reader.py` / `gguf_writer.py` — IO helpers for reading/writing GGUF tensors and metadata.
- `metadata.py` / `constants.py` / `tensor_mapping.py` — Schema definitions and tensor routing helpers.
- `quants.py` / `quick_4bits_ops.py` — Quantization utilities.
- `lazy.py` / `utility.py` — Lazy-loading helpers and shared GGUF utilities.
- `vocab.py` — Vocabulary parsing for GGUF tokenizers.

## Notes
- Keep GGUF support centralized here so runtimes/engines can depend on a single implementation.
- When updating GGUF spec support, document changes and align with WAN22 runtime expectations.
