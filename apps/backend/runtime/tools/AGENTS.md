# apps/backend/runtime/tools Overview
Date: 2025-12-31
Owner: Runtime Maintainers
Last Review: 2026-01-02
Status: Active

## Purpose
- Backend runtime “tools” that perform heavyweight offline-style operations (e.g. converting checkpoints) and are exposed via `/api/tools/*`.

## Key Files
- `apps/backend/runtime/tools/gguf_converter.py` — Converts SafeTensors (including sharded `*.safetensors.index.json`) to GGUF with quantization + verification.

## Notes
- Tools should be deterministic, auditable, and fail loud (no silent fallbacks).
- When adding metadata to GGUF outputs, prefer stable keys and avoid leaking absolute local filesystem paths.
- 2026-01-02: Added standardized file header docstrings to the tools facade (`__init__.py`) (doc-only change; part of rollout).
