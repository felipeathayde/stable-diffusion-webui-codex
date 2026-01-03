# apps/backend/runtime Overview
<!-- tags: backend, runtime, overview -->
Date: 2025-10-30
Owner: Runtime Maintainers
Last Review: 2026-01-02
Status: Active

## Purpose
- Provides reusable runtime components shared across engines: attention kernels, adapters, text processing, memory policies, sampling utilities, model loaders, and model-specific runtimes (SD, Flux, Chroma, WAN22).

## Key Subdirectories
- `attention/` — Attention backends and related kernels.
- `adapters/` — Runtime adapters (e.g., LoRA, SafeTensors helpers).
- `model_parser/` — Codex-native checkpoint parser plans and conversions replacing `huggingface_guess`.
- `text_processing/` — Tokenization, prompt parsing, and textual inversion helpers.
- `sampling/` — Sigma builders, schedulers, Philox integration, and sampling drivers.
- `memory/` — VRAM/CPU memory policies and management helpers.
- `ops/` — Low-level tensor operations leveraged by engines.
- `models/` — Model registry/load helpers (checkpoints, VAEs, etc.).
- `{sd, flux, chroma, wan22}/` — Model/runtime specific implementations.
- `vision/` — Vision encoder runtimes (clip specs/registry/encoders) shared across engines and patchers.
- `processing/` — High-level input preprocessing utilities shared by use cases.
- `workflows/` — Shared orchestration helpers for Codex generation workflows (txt2img, img2img, video).
- `common/` — Shared building blocks (e.g., core (UNet/DiT) wrappers) used across runtimes.
- `misc/` — Smaller helper modules that don’t fit other buckets (logging, strict checks, etc.).
- `modules/` — Compatibility wrappers expected by legacy callers (will shrink over time).
- `kernels/` — Custom CUDA/C++ kernels where required.

## Notes
- Keep runtime logic model-agnostic when possible; place model-specific code under the dedicated `{model}/` folders.
- Avoid duplicating helpers across engines—centralize them here to maintain parity.
- 2025-12-03: Processing models expose `RefinerConfig` and carry refiner configs on `CodexProcessingTxt2Img`/`CodexHighResConfig` (global + hires) for stage-based refiner execution.
- 2025-12-03: Sampler driver checks `backend_state.should_stop` each step and honors `/api/tasks/{id}/cancel` (immediate) by raising `RuntimeError("cancelled")` to abort sampling.
- 2025-11-03: New `runtime.call_trace` module exposes global function-call tracing via `enable()/disable()` and `enable_from_env()`. The API entrypoint wires this behind `--trace-debug`/`CODEX_TRACE_DEBUG` and logs each Python function call at `DEBUG` using the `backend.calltrace` logger. As of 2025-11-14, only modules under `apps.*` are recorded to avoid 3rd-party flood, and each function logs at most 10 calls by default (override via `--trace-debug-max-per-func N`, `N<=0` disables the cap).
- 2025-11-04: Added streaming materialization helpers to `runtime.utils.FilterPrefixView`/`LazySafetensorsDict` so safetensor-backed parser components load with a single handle instead of reopening per key (prevents Windows `torch_cpu.dll` crashes during SDXL parsing).
- 2025-11-14: Weight/bias fetch logs under `runtime.ops.operations` are now rate-limited via `CODEX_WEIGHT_FETCH_LOG_LIMIT` (default 10 per layer class). Set to `0` to disable the log entirely or raise when diagnosing dtype/offload issues.
- 2025-11-25: SDXL CLIP converters now handle OpenCLIP BigG resblock layouts without the double-`transformer` prefix bug; CLIP-G keys under `transformer.resblocks.*` are normalized to `transformer.text_model.encoder.*` (preserving `logit_scale`) so validations no longer warn on missing `layer_norm1`.
- 2025-12-15: `runtime/tools/gguf_converter.py` emits real quantized GGUF using the shared GGUF writer + quant kernels and streams tensor data instead of buffering entire checkpoints in memory.
- 2025-12-19: GGUF converter quant menu expanded to include `Q2_K/Q3_K/IQ4_NL`, mixed schemes (`Q4_K_M/Q5_K_M`), per-tensor override rules, and legacy `Q4_0/Q4_1/Q5_0/Q5_1/Q6_K` (in addition to `Q8_0/Q5_K/Q4_K`).
- 2025-12-30: GGUF converter now supports sharded SafeTensors inputs via `*.safetensors.index.json` (or by pointing at a directory containing the index); no manual shard merge required.
- 2025-12-29: Sampling and utils now avoid importing heavy runtime ops/quantization at module import time (keeps API startup and `/api/models`/QuickSettings paths scans lightweight).
- 2025-12-29: Runtime exception logging now prefers `CODEX_ROOT/logs` when `CODEX_ROOT` is set (prevents CWD-dependent log placement).
- 2026-01-01: GGUF checkpoint loader supports opt-in load-time dequantization via `--gguf-dequantize-upfront` (otherwise weights dequantize on the fly).
- 2026-01-01: Live preview utilities now live in `runtime/live_preview.py` (method enum, preview decode helper, and debug preview-factor fitting/logging) so workflows and API layers don’t duplicate preview logic.
- 2026-01-02: Added standardized file header docstrings to runtime modules (doc-only change; part of rollout).
- 2026-01-02: Added standardized file header docstrings to runtime package scaffolding (`__init__.py`, `pipeline_debug.py`, `shared.py`, `trace.py`) (doc-only change; part of rollout).
