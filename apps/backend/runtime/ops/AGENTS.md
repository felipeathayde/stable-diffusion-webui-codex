# apps/backend/runtime/ops Overview
Date: 2025-10-30
Owner: Runtime Maintainers
Last Review: 2026-01-18
Status: Active

## Purpose
- Tensor operations (custom matmul, fused ops, etc.) leveraged by engines and runtimes.

## Notes
- Introduce new ops here and document their expected inputs/outputs to keep usages consistent.
- `operations_bnb.py` now exposes a `BnbQuantConfig` + registry so downstream loaders request 4bit helpers without importing bitsandbytes internals; register additional quant types in the registry (and update documentation) when new variants land.
- `ops/__init__.py` lazy facade only treats `bitsandbytes` as optional; unexpected import failures now surface loudly to avoid hiding real bugs.
- 2025-12-13: `CodexOperationsGGUF` now supports GGUF-style state dict loading for `Linear`/`Embedding` plus Conv/Norm variants (`Conv{1,2,3}d`, `ConvTranspose{1,2,3}d`, `GroupNorm`, `LayerNorm`) so nn.Module runtimes (ex.: WAN22) can load GGUF weights without model-specific runners.
- 2026-01-01: GGUF CPU LRU cache is guarded to CPU-resident weights only (prevents unintended CPU->GPU transfers when running on CUDA).
- 2026-01-02: Added standardized file header docstrings to ops facades and GGUF runtime helpers (doc-only change; part of rollout).
- 2026-01-29: Added `ops/codexpack_cuda.py` to best-effort load the `codexpack_cuda` extension (prebuilt or in-place build) for CodexPack packed GGUF execution.
