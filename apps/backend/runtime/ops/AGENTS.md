# apps/backend/runtime/ops Overview
Date: 2025-10-30
Last Review: 2026-02-09
Status: Active

## Purpose
- Tensor operations (custom matmul, fused ops, etc.) leveraged by engines and runtimes.

## Notes
- Introduce new ops here and document their expected inputs/outputs to keep usages consistent.
- NF4/FP4 4-bit integration was removed: NF4/FP4 checkpoints are **not supported** and must fail loud with an actionable error (convert to GGUF or use safetensors fp16/bf16/fp32).
- `using_codex_operations(..., weight_format="gguf")` selects GGUF-aware torch.nn op shims; any other `weight_format` must raise `NotImplementedError` (no silent fallback).
- 2025-12-13: `CodexOperationsGGUF` now supports GGUF-style state dict loading for `Linear`/`Embedding` plus Conv/Norm variants (`Conv{1,2,3}d`, `ConvTranspose{1,2,3}d`, `GroupNorm`, `LayerNorm`) so nn.Module runtimes (ex.: WAN22) can load GGUF weights without model-specific runners.
- 2026-01-01: GGUF CPU LRU cache is guarded to CPU-resident weights only (prevents unintended CPU->GPU transfers when running on CUDA).
- 2026-01-02: Added standardized file header docstrings to ops facades and GGUF runtime helpers (doc-only change; part of rollout).
- 2026-01-29: Added `ops/codexpack_cuda.py` to best-effort load the `codexpack_cuda` extension (prebuilt or in-place build) for CodexPack packed GGUF execution.
- 2026-02-09: Removed inference-tensor materialization mitigations for the version-counter crash; fixes are scoped to correct request entrypoints (`get_learned_conditioning` uses `torch.no_grad()` instead of `torch.inference_mode()`).
