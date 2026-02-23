# apps/backend/runtime/ops Overview
Date: 2025-10-30
Last Review: 2026-02-22
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
- 2026-02-15: `CodexOperationsGGUF.Embedding` constructor is now lazy for standard GGUF paths (`_weight` absent): it builds metadata on `meta` device and defers real weight materialization to `_load_from_state_dict`, preventing large eager embedding allocations before GGUF state-dict load. Explicit `_weight` constructor path remains eager/usable (no dummy placeholder), and strict missing-key loads now report `weight` as missing.
- 2026-02-15: GGUF `Embedding.forward` now enforces floating compute dtype (`weight.computation_dtype` when available; fallback `fp16` on CUDA / `fp32` on CPU) before `F.embedding`, preventing integer (`Byte`) activations from leaking into GGUF `Linear` dense matmul. GGUF `Linear.forward` now fails loud on non-floating activations with an actionable contract error.
- 2026-02-16: GGUF `Linear._load_from_state_dict(...)` now normalizes loaded dense weights for HF UMT5 compatibility:
  - packed `CodexParameter` weights with `uint8` storage are reinterpreted to `int8` storage (bytes preserved) so HF FFN gates do not cast activations to `uint8`,
  - non-quantized integer dense weights are cast to computation dtype at load time.
  This fixes root-cause `addmm_cuda ... Byte` failures in WAN22 GGUF TE paths (instead of only guard-level fail-loud behavior).
- 2026-02-22: `using_codex_operations(...)` now uses context-local operation context (`ContextVar`) and a reentrant lock around global torch.nn patch windows, preventing cross-request context bleed and overlapping monkeypatch races under concurrent execution.
- 2026-02-22: Removed GGUF run-scoped dequant-forward cache API (`enable/disable/is_enabled` and lvl1/lvl2 internals) from `operations_gguf.py`; forward path now always uses direct move+dequant behavior while CPU LRU cache policy (`none|cpu_lru`) remains supported.
