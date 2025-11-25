# apps/backend/runtime Overview
Date: 2025-10-30
Owner: Runtime Maintainers
Last Review: 2025-10-31
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
- 2025-11-03: New `runtime.call_trace` module exposes global function-call tracing via `enable()/disable()` and `enable_from_env()`. The API entrypoint wires this behind `--trace-debug`/`CODEX_TRACE_DEBUG` and logs each Python function call at `DEBUG` using the `backend.calltrace` logger. As of 2025-11-14, only modules under `apps.*` are recorded to avoid 3rd-party flood, and each function logs at most 10 calls by default (override via `--trace-debug-max-per-func N`, `N<=0` disables the cap).
- 2025-11-04: Added streaming materialization helpers to `runtime.utils.FilterPrefixView`/`LazySafetensorsDict` so safetensor-backed parser components load with a single handle instead of reopening per key (prevents Windows `torch_cpu.dll` crashes during SDXL parsing).
- 2025-11-14: Weight/bias fetch logs under `runtime.ops.operations` are now rate-limited via `CODEX_WEIGHT_FETCH_LOG_LIMIT` (default 10 per layer class). Set to `0` to disable the log entirely or raise when diagnosing dtype/offload issues.
- 2025-11-25: SDXL CLIP converters now handle OpenCLIP BigG resblock layouts without the double-`transformer` prefix bug; CLIP-G keys under `transformer.resblocks.*` are normalized to `transformer.text_model.encoder.*` so validations no longer warn on missing `layer_norm1`.
