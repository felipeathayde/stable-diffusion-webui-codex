# AGENT — Runtime Checkpoint IO

Purpose: Checkpoint IO helpers used by the runtime (safetensors / GGUF / guarded pickle).

Key files:
- `apps/backend/runtime/checkpoint/io.py`: `load_torch_file`, GGUF loaders, and config read helpers.
- `apps/backend/runtime/checkpoint/safetensors_header.py`: SafeTensors header-only readers (incl. primary dtype hint) for lightweight tooling/logging.

Notes:
- Keep checkpoint IO lightweight and avoid importing heavy runtime modules at import time.
- Favor SafeTensors and safe torch loading where possible; only fall back to guarded pickle when explicitly allowed.
- 2026-01-28: `io.py` now also exposes `read_gguf_metadata(...)` so engines can inspect trusted GGUF provenance without importing `apps.backend.quantization.*` directly (import guardrail compliance).

Last Review: 2026-01-28
