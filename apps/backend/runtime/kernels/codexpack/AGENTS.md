<!-- tags: backend, runtime, kernels, cuda, codexpack -->
# apps/backend/runtime/kernels/codexpack Overview
Date: 2026-01-29
Last Review: 2026-02-18
Status: Active

## Purpose
- SM86+ CUDA extension for CodexPack packed GGUF execution.
- Provides `torch.ops.codexpack.q4k_tilepack_linear(...)` for `cuda.ggml_q4_k.linear.tilepack_v1` packed weights.

## Build
Prerequisites:
- Python in the repo venv (`$CODEX_ROOT/.venv`).
- PyTorch CUDA build installed (not `+cpu`).
- CUDA toolkit with `nvcc` available on PATH (CUDA 12.x recommended).

In-place build (development):
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)"
cd "$CODEX_ROOT/apps/backend/runtime/kernels/codexpack"
PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" setup.py build_ext --inplace
```

Correctness harness (SM86+; after build):
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)"
PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" \
  apps/backend/runtime/kernels/codexpack/validate_q4k_tilepack_linear.py
```

Notes:
- Runtime policy is SM86+ only. The kernel will fail loud on older GPUs even if it compiles.
- This extension is not auto-downloaded. When missing, CodexPack execution errors with clear build/install guidance.
- Optional (dev-only): set `CODEX_CODEXPACK_JIT=1` to allow the runtime to attempt a local JIT build if the extension is missing (requires `nvcc`).
- 2026-02-18: CLI success output in `validate_q4k_tilepack_linear.py` now emits via `apps.backend.infra.stdio.write_stdout(...)` (contract-compatible stdout line, no direct `print(...)` callsite).

## Files
- `setup.py` — Build script (`CUDAExtension`).
- `codexpack_binding.cpp` — Torch op registration (`TORCH_LIBRARY(codexpack, ...)`).
- `q4k_tilepack_linear.cu` — CUDA implementation (correctness-first; optimize later).
- `validate_q4k_tilepack_linear.py` — CUDA correctness harness (packed op vs dequant+linear baseline).
