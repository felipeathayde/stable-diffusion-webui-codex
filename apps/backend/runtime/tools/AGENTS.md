# apps/backend/runtime/tools Overview
Date: 2025-12-31
Owner: Runtime Maintainers
Last Review: 2026-01-13
Status: Active

## Purpose
- Backend runtime “tools” that perform heavyweight offline-style operations (e.g. converting checkpoints) and are exposed via `/api/tools/*`.

## Key Files
- `apps/backend/runtime/tools/gguf_converter.py` — Converts SafeTensors (including sharded `*.safetensors.index.json`) to GGUF with quantization + verification.
- `apps/backend/runtime/tools/gguf_converter_key_mapping.py` — Hugging Face → GGUF tensor-name remapping helpers (layer-indexed mappings).
- `apps/backend/runtime/tools/gguf_converter_safetensors_source.py` — SafeTensors source helpers (single-file + sharded index/dir).
- `apps/backend/runtime/tools/gguf_converter_quantization.py` — Quantization selection + override rules for the converter.
- `apps/backend/runtime/tools/gguf_converter_tensor_planner.py` — Tensor conversion planning helpers (types + stored byte shapes).
- `apps/backend/runtime/tools/gguf_converter_types.py` — Public converter types (config, quantization enum, progress, verification error).
- `apps/backend/runtime/tools/gguf_converter_metadata.py` — GGUF metadata injection helpers (provenance + arch keys).
- `apps/backend/runtime/tools/gguf_converter_verify.py` — GGUF output verification helpers (tensor tables + spot-checks).

## Notes
- Tools should be deterministic, auditable, and fail loud (no silent fallbacks).
- When adding metadata to GGUF outputs, prefer stable keys and avoid leaking absolute local filesystem paths.
- 2026-01-13: GGUF converter metadata no longer writes `general.author` or `general.source.repo_url` (keep provenance in `general.repo_url`/`general.version` and `general.source.url`).
- 2026-01-02: Added standardized file header docstrings to the tools facade (`__init__.py`) (doc-only change; part of rollout).
