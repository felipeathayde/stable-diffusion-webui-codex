<!-- tags: backend, codex, quantization, gguf -->
# apps/backend/quantization Overview
Date: 2025-12-15
Owner: Runtime Maintainers
Last Review: 2025-12-15
Status: Active

## Purpose
- Canonical implementation for GGUF quantized tensors in Codex runtime: `CodexParameter` storage invariants, (de)quant kernels, and GGUF IO helpers (reader/writer/constants) used by tools and runtimes.

## Key Files
- `core.py` — Quant enums/registry (`QuantType` = `GGMLQuantizationType`, `QuantSpec`, `register_quant`).
- `tensor.py` — `CodexParameter` / storage invariants for packed GGUF tensors.
- `api.py` — Public API (`dequantize`, `bake`, `quantize`).
- `gguf_loader.py` — GGUF → state_dict loader (used by Z Image GGUF text encoder).
- `kernels/` — Kernel registry bootstrap (ported dequant blocks; add quantize here when needed).
- `gguf/` — GGUF format IO (reader/writer/constants). This replaces the old `apps/backend/gguf/` package.
- `apps/backend/runtime/ops/operations_gguf.py` — runtime integration (dequantize helper + optional CPU LRU cache knobs).

## Notes
- This package is the only place that should own GGUF quantization + IO. Do not reintroduce `apps/backend/gguf/` (deprecated and removed).
- Quantized tensors are byte-packed; do not cast storage dtypes. Only `computation_dtype` controls dequant output dtype.
