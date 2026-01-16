# apps/backend/runtime/tools Overview
Date: 2025-12-31
Owner: Runtime Maintainers
Last Review: 2026-01-16
Status: Active

## Purpose
- Backend runtime “tools” that perform heavyweight offline-style operations (e.g. converting checkpoints) and are exposed via `/api/tools/*`.

## Key Files
- `apps/backend/runtime/tools/gguf_converter.py` — Converts SafeTensors (including sharded `*.safetensors.index.json`) to GGUF with quantization + verification.
- `apps/backend/runtime/tools/gguf_converter_specs.py` — Typed converter specs (profiles/layouts + quantization policy rule types).
- `apps/backend/runtime/tools/gguf_converter_profiles.py` — Profile registry: resolves layout/planner/key mapping + per-model dtype policies.
- `apps/backend/runtime/tools/gguf_converter_model_metadata.py` — Lists vendored model metadata (org/repo + supported components/config dirs) for the GGUF converter UI.
- `apps/backend/runtime/tools/gguf_converter_float_groups.py` — Defines profile-scoped FP16/FP32 float dtype groups exposed as UI knobs.
- `apps/backend/runtime/tools/gguf_converter_key_mapping.py` — Hugging Face → GGUF tensor-name remapping helpers (layer-indexed mappings).
- `apps/backend/runtime/tools/gguf_converter_safetensors_source.py` — SafeTensors source helpers (single-file + sharded index/dir).
- `apps/backend/runtime/tools/gguf_converter_quantization.py` — Quantization selector + generic per-tensor shape/block compatibility rules.
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
- 2026-01-14: Fixed `concat_dim0` streaming writes to allow variable dim0 sizes (required by Flux single-block `linear1` fusion: q/k/v + `proj_mlp`).
- 2026-01-14: Flux GGUF quantization now keeps sensitive IO projection weights in float (F32/F16) and keeps Flux 1D tensors in F32 (biases + norm scales), matching known-good community files and preventing residual noise regressions.
- 2026-01-14: GGUF converter dispatch is now profile-driven (typed registry): model-specific dtype “overrides” are formalized as per-model quantization policies (user `tensor_type_overrides` remain supported, but policy rules can be marked required).
- 2026-01-14: Follow-up: fixed a missing `GGUFKeyLayout` import in `gguf_converter.py` introduced during the profile-registry refactor (NameError at runtime).
- 2026-01-15: Removed a stale Flux planner dtype override injection that imported a deleted type; Flux dtype rules live in the profile quantization policy.
- 2026-01-15: Flux mixed presets (`Q5_K_M` / `Q4_K_M`) now keep additional IO weights in F32 (larger GGUF, higher quality).
- 2026-01-15: GGUF converter now supports explicit `profile_id` selection (UI can avoid heuristics) and a vendored preset list for picking configs.
- 2026-01-15: GGUF converter exposes FP16/FP32 via profile-scoped float groups (Advanced) for selected tensors (per-profile patterns).
- 2026-01-16: Replaced Flux-only dtype knobs with generic profile-scoped FP16/FP32 float groups (applies to any supported converter profile).
- 2026-01-16: Vendored selector now uses “model metadata” (org/repo + component) rather than listing raw config-dir presets.
- 2026-01-16: Vendored model metadata scanner no longer classifies `*ForCausalLM` configs as converter components and labels diffusion components as `denoiser` for UI display.
- 2026-01-02: Added standardized file header docstrings to the tools facade (`__init__.py`) (doc-only change; part of rollout).
