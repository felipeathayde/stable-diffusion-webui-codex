# AGENT — Runtime Checkpoint IO

Purpose: Checkpoint IO helpers used by the runtime (safetensors / GGUF / guarded pickle).

Key files:
- `apps/backend/runtime/checkpoint/io.py`: `load_torch_file`, GGUF loaders, and config read helpers.

Notes:
- Keep checkpoint IO lightweight and avoid importing heavy runtime modules at import time.
- Favor SafeTensors and safe torch loading where possible; only fall back to guarded pickle when explicitly allowed.

Last Review: 2026-01-18

