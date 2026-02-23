<!-- tags: backend, codex, quantization, gguf -->
# apps/backend/quantization Overview
Date: 2025-12-15
Last Review: 2026-02-15
Status: Active

## Purpose
- Canonical implementation for GGUF quantized tensors in Codex runtime: `CodexParameter` storage invariants, (de)quant kernels, GGUF IO helpers (reader/writer/constants), and CodexPack packed-weight containers used by (future) packed-kernel execution.

## Key Files
- `core.py` — Quant enums/registry (`QuantType` = `GGMLQuantizationType`, `QuantSpec`, `register_quant`).
- `tensor.py` — `CodexParameter` / storage invariants for packed GGUF tensors.
- `codexpack_keymaps.py` — CodexPack `keymap_id` registry (fail loud on unknown mappings).
- `codexpack_tensor.py` — CodexPack packed-weight container(s) for runtime execution (e.g., packed Q4_K linear blobs).
- `api.py` — Public API (`dequantize`, `bake`, `quantize`).
- `gguf_loader.py` — GGUF → state_dict loader (includes CodexPack auto-detection and packed-weight surfacing).
- `kernels/` — Kernel registry bootstrap (ported dequant blocks; add quantize here when needed).
- `gguf/` — GGUF format IO (reader/writer/constants). This replaces the old `apps/backend/gguf/` package.
- `apps/backend/runtime/ops/operations_gguf.py` — runtime integration (dequantize helper + optional CPU LRU cache knobs).

## Notes
- This package is the only place that should own GGUF quantization + IO. Do not reintroduce `apps/backend/gguf/` (deprecated and removed).
- Quantized tensors are byte-packed; do not cast storage dtypes. Only `computation_dtype` controls dequant output dtype.
- CodexPack `*.codexpack.gguf` is a Codex-specific packed GGUF contract (see `apps/backend/quantization/gguf/codexpack.py` for strict parsing/validation). Runtime must fail loud on unknown `keymap_id` / missing prerequisites.
- 2025-12-19: Tooling gained additional NumPy quant packers (`Q2_K/Q3_K/IQ4_NL` + `Q4_0/Q4_1/Q5_0/Q5_1/Q6_K`) so the GGUF Converter can emit more GGML types.
- 2026-01-01: `CodexParameter.to(...)` avoids unnecessary clones for packed GGUF tensors and returns `self` for no-op device moves (reduces per-step overhead and wrapper churn).
- 2026-01-02: Added standardized file header docstrings across quantization modules (public API, registry/cache/ops, GGUF loader, kernels bootstrap, GGUF IO helpers) (doc-only change; part of rollout).
- 2026-01-03: Standardized upstream references in quantization docs/comments (avoid naming unrelated projects as baselines; keep necessary attributions in headers/notices).
- 2026-01-19: Made `apps.backend.quantization` import-light (lazy facade) so `apps.backend.quantization.gguf` can be torch-free on import; kernel registration now happens when importing `apps.backend.quantization.api`.
- 2026-01-20: Removed unreferenced legacy quantization helpers (`quantization/ops.py` and `gguf/{lazy,metadata,tensor_mapping,utility,vocab}.py`).
- 2026-02-15: `tensor.py` and `codexpack_tensor.py` now use explicit NumPy->tensor no-copy helpers (readonly warning suppressed intentionally) to avoid eager host copies for packed GGUF/CodexPack weights.
- 2026-02-23: `gguf_loader.py::_resolve_target_device(...)` now defaults to memory-manager mount-device authority when no device is provided, removing an implicit CPU default in GGUF load paths.
