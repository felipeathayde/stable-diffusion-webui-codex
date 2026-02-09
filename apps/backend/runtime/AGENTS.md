# apps/backend/runtime Overview
<!-- tags: backend, runtime, overview -->
Date: 2025-10-30
Last Review: 2026-02-09
Status: Active

## Purpose
- Provides reusable runtime components shared across engines: attention kernels, adapters, text processing, memory policies, sampling utilities, model loaders, and model-family runtimes (SD, Flux, Chroma, ZImage, WAN22).

## Key Subdirectories
- `attention/` — Attention backends and related kernels.
- `adapters/` — Runtime adapters (e.g., LoRA, SafeTensors helpers).
- `diagnostics/` — Tracing/timeline/debug helpers (optional; used for diagnosis).
- `model_parser/` — Codex-native checkpoint parser plans and conversions replacing `huggingface_guess`.
- `text_processing/` — Tokenization, prompt parsing, and textual inversion helpers.
- `sampling/` — Sigma builders, schedulers, Philox integration, and sampling drivers.
- `memory/` — VRAM/CPU memory policies and management helpers.
- `ops/` — Low-level tensor operations leveraged by engines.
- `checkpoint/` — Checkpoint IO helpers (safetensors/GGUF/pickle + config reads).
- `state_dict/` — Lightweight state-dict views + small state-dict utilities.
- `models/` — Model registry/load helpers (checkpoints, VAEs, etc.).
- `families/` — Model/runtime-specific implementations by engine family (`sd/`, `flux/`, `chroma/`, `zimage/`, `wan22/`).
- `vision/` — Vision encoder runtimes (clip specs/registry/encoders) shared across engines and patchers.
- `processing/` — High-level input preprocessing utilities shared by use cases.
- `pipeline_stages/` — Shared pipeline helper stages consumed by canonical use-cases (Option A; no engine-specific pipelines).
- `streaming/` — Shared segment streaming controller primitives used by multiple family streaming wrappers (Flux/WAN22).
- `common/` — Shared building blocks (e.g., core (UNet/DiT) wrappers) used across runtimes.
- `misc/` — Smaller helper modules that don’t fit other buckets (logging, strict checks, etc.).
- `sampling_adapters/` — Sampling adapter wrappers used by samplers/patchers.
- `kernels/` — Custom CUDA/C++ kernels where required.

## Notes
- Keep runtime logic model-agnostic when possible; place model-specific code under `families/<family>/`.
- Avoid duplicating helpers across engines—centralize them here to maintain parity.
- 2026-01-24: Attention backend selection is now driven by the runtime memory config (seeded from `/api/options` key `codex_attention_backend` at bootstrap, and switchable at runtime via `memory_management.set_attention_backend(...)`).
- Runtime layout: family runtimes live under `apps/backend/runtime/families/<family>/`; keep `apps/backend/runtime/` for generic runtime modules shared across families (plan: `.sangoi/plans/2026-01-17-backend-runtime-families-layout.md`).
- 2025-12-03: Processing models expose `RefinerConfig` and carry refiner configs on `CodexProcessingTxt2Img`/`CodexHiresConfig` (global + hires) for stage-based refiner execution.
- 2026-02-08: `RefinerConfig` pointer semantics now use `swap_at_step` (serialized as `switch_at_step`) instead of refiner step counts.
- 2025-12-03: Sampler driver checks `backend_state.should_stop` each step and honors `/api/tasks/{id}/cancel` (immediate) by raising `RuntimeError("cancelled")` to abort sampling.
- 2025-11-03: `runtime.diagnostics.call_trace` exposes global function-call tracing via `enable()/disable()` and `enable_from_env()`. The API entrypoint wires this behind `--trace-debug`/`CODEX_TRACE_DEBUG` and logs each Python function call at `DEBUG` using the `backend.calltrace` logger. As of 2025-11-14, only modules under `apps.*` are recorded to avoid 3rd-party flood, and each function logs at most 10 calls by default (override via `--trace-debug-max-per-func N`, `N<=0` disables the cap).
- 2025-11-04: Added streaming materialization helpers to `runtime.state_dict.views.FilterPrefixView`/`LazySafetensorsDict` so safetensor-backed parser components load with a single handle instead of reopening per key (prevents Windows `torch_cpu.dll` crashes during SDXL parsing).
- 2025-11-14: Weight/bias fetch logs under `runtime.ops.operations` are now rate-limited via `CODEX_WEIGHT_FETCH_LOG_LIMIT` (default 10 per layer class). Set to `0` to disable the log entirely or raise when diagnosing dtype/offload issues.
- 2025-11-25: SDXL CLIP converters now handle OpenCLIP BigG resblock layouts without the double-`transformer` prefix bug; CLIP-G keys under `transformer.resblocks.*` are normalized to `transformer.text_model.encoder.*` (preserving `logit_scale`) so validations no longer warn on missing `layer_norm1`.
- 2025-12-15: `runtime/tools/gguf_converter.py` emits real quantized GGUF using the shared GGUF writer + quant kernels and streams tensor data instead of buffering entire checkpoints in memory.
- 2025-12-19: GGUF converter quant menu expanded to include `Q2_K/Q3_K/IQ4_NL`, mixed schemes (`Q4_K_M/Q5_K_M`), per-tensor override rules, and legacy `Q4_0/Q4_1/Q5_0/Q5_1/Q6_K` (in addition to `Q8_0/Q5_K/Q4_K`).
- 2025-12-30: GGUF converter now supports sharded SafeTensors inputs via `*.safetensors.index.json` (or by pointing at a directory containing the index); no manual shard merge required.
- 2025-12-29: Sampling and utils now avoid importing heavy runtime ops/quantization at module import time (keeps API startup and `/api/models`/QuickSettings paths scans lightweight).
- 2025-12-29: Runtime exception logging now prefers `CODEX_ROOT/logs` when `CODEX_ROOT` is set (prevents CWD-dependent log placement).
- 2026-01-31: `CODEX_LOG_FILE` now attaches a file handler to the `backend` logger hierarchy as well as the root logger, since `backend.propagate=False` would otherwise yield an empty log file for backend logs (launcher “Write to log file”).
- 2026-01-01: GGUF checkpoint loader supports opt-in load-time dequantization via `--gguf-exec=dequant_upfront` (otherwise weights dequantize on the fly via `dequant_forward`).
- 2026-01-23: Started gating future GGUF packed-kernel execution via `--gguf-exec=cuda_pack` (reserved; fail loud until implemented) and introduced `--lora-online-math` to make online LoRA semantics explicit.
- 2026-01-04: Added `runtime.checkpoint.io.load_gguf_state_dict(...)` as the canonical GGUF load wrapper so runtime codepaths honor global GGUF flags consistently (no direct loader calls).
- 2026-01-18: `runtime.checkpoint.io.load_gguf_state_dict(...)` supports explicit GGUF dequantization policy (`dequantize` + `computation_dtype`) so callers can centralize GGUF loads without importing `apps.backend.quantization.*` directly.
- 2026-01-01: Live preview utilities now live in `runtime/live_preview.py` (method enum, preview decode helper, and debug preview-factor fitting/logging) so workflows and API layers don’t duplicate preview logic.
- 2026-01-02: Added standardized file header docstrings to runtime modules (doc-only change; part of rollout).
- 2026-01-02: Added standardized file header docstrings to runtime package scaffolding (`__init__.py` and diagnostics modules) (doc-only change; part of rollout).
- 2026-01-03: Standardized upstream references in runtime docs/comments to prefer Hugging Face Diffusers as the behaviour baseline.
- 2026-02-09: Version-counter mitigation is handled at engine conditioning entrypoints (`torch.no_grad()`); runtime no longer carries inference-tensor materialization shims for this class of failure.
