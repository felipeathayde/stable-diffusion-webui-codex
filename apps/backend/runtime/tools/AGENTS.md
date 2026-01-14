# apps/backend/runtime/tools Overview
Date: 2025-12-31
Owner: Runtime Maintainers
Last Review: 2026-01-14
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
- 2026-01-13: GGUF converter metadata uses a Codex UI schema (`model.*`, `codex.*`, `gguf.*`) and avoids verbose conversion input keys (`codex.source_*`).
- 2026-01-13: GGUF converter supports cooperative cancellation (Tools API cancel flag) and the tools API defaults to no-overwrite when the output file already exists.
- 2026-01-13: GGUF converter supports Flux transformer planning: maps Diffusers `FluxTransformer2DModel` keys (`transformer_blocks.*`, `single_transformer_blocks.*`) into the Comfy/Codex Flux runtime layout (`double_blocks.*`, `single_blocks.*`, `img_in`, `txt_in`, `time_in`, `vector_in`, `guidance_in`, `final_layer`).
- 2026-01-14: GGUF converter now supports a `comfy_layout` toggle: for Flux/ZImage transformer exports, when enabled it maps Diffusers keys into the Comfy/Codex runtime layout (`double_blocks.*`, `single_blocks.*`); when disabled it preserves source key names. Output records `codex.converter.comfy_layout` in metadata.
- 2026-01-02: Added standardized file header docstrings to the tools facade (`__init__.py`) (doc-only change; part of rollout).
