# apps/backend/gguf Overview
Date: 2025-10-30
Owner: Runtime Maintainers
Last Review: 2025-10-30
Status: Active

## Purpose
- Houses utilities for working with GGUF-formatted checkpoints (metadata parsing, tensor mapping, quantization helpers) used by WAN22 and other GGUF-enabled engines.

## Key Files
- `gguf_reader.py` / `gguf_writer.py` — IO helpers for reading/writing GGUF tensors and metadata.
- `metadata.py` / `constants.py` / `tensor_mapping.py` — Schema definitions and tensor routing helpers.
- `quants/` package — Quantization registry (`registry.py`), helpers (`utils.py`), and kernel families.
- `quants/kernels/base/forge_*.py` — Forge-derived BF16/Q4/Q5/Q8 math preserved under MIT license headers.
- `quants/kernels/k_family/forge_*.py` — Forge-derived Q2_K–Q6_K families, imported verbatim per kernel with MIT attribution.
- `quick_4bits_ops.py` — Shared bit-packing helpers used by kernel bake paths.
- `lazy.py` / `utility.py` — Lazy-loading helpers and shared GGUF utilities.
- `vocab.py` — Vocabulary parsing for GGUF tokenizers.

## Notes
- Keep GGUF support centralized here so runtimes/engines can depend on a single implementation.
- When updating GGUF spec support, document changes and align with WAN22 runtime expectations.
- The quantization stack is moving to `QuantKernel` registry abstractions; new kernels should register through the shared helpers rather than introducing standalone classes.
- Forge-derived low-level kernels now live in `quants/kernels/base/forge_*.py` (base types) and `quants/kernels/k_family/forge_*.py` (Q2_K–Q6_K); keep them immutable except for upstream syncs and add new math in Codex-specific wrappers.
