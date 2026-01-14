# apps/backend/engines/common Overview
Date: 2025-10-28
Owner: Engine Maintainers
Last Review: 2026-01-14
Status: Active

## Purpose
- Shared engine utilities (base classes, mixins, helpers) reused across SD, Flux, Chroma, WAN22 engines.

## Notes
- `CodexDiffusionEngine` now subclasses `BaseInferenceEngine`; implement `_build_components(bundle, *, options)` to assemble runtime objects during `load()`.
- Engines receive pre-materialised `DiffusionModelBundle` instances; avoid invoking legacy loaders inside subclasses.
- Model family flags (`is_sd1`, `is_sd2`, `is_sd3`, `is_sdxl`) remain read-only; call `register_model_family(...)` inside `_build_components` after deriving the runtime.
- Lifecycle hooks: `_on_unload()` lets subclasses clear caches, while `status()` now reports `model_ref`, bundle source, and registered families.
- Tiling/CFG scale toggles remain available but emit structured logs when changed.
- 2025-12-30: `CodexDiffusionEngine.load()` now calls `unload()` when already loaded, and `unload()` clears bound components from the memory manager (prevents duplicate model instances accumulating when reloading with different overrides).
- 2026-01-01: Engine docs note that `get_learned_conditioning(...)` may return either a dict or a cross-attn tensor (both supported by `compile_conditions`).
- 2026-01-02: Added standardized file header docstrings to shared engine modules (doc-only change; part of rollout).
- 2026-01-04: `CodexObjects` renamed `unet` → `denoiser`; engines store their sampling core patcher under `codex_objects.denoiser` (UNet for SD-family; transformer/DiT for Flux/Z-Image/WAN).
- 2026-01-06: VAE override (`vae_path`) now unwraps wrapper VAEs via `first_stage_model` before calling `safe_load_state_dict` (fixes `'VAE' object has no attribute 'state_dict'`).
- 2026-01-06: VAE/TE selection is explicit via `vae_source`/`tenc_source` + paths; core-only `.gguf` checkpoints never treat these paths as state-dict overrides, and ZImage always treats them as external selection (may be dir/gguf).
- 2026-01-06: Refreshed `base.py` file header blocks to document `vae_source`/`tenc_source` validation and core-only `.gguf` semantics (doc-only change).
- 2026-01-06: Generation metadata no longer falls back to `"Automatic"` for sampler/scheduler; missing values serialize as null to surface invalid inputs.
- 2026-01-08: `base.py` now imports `TextEncoderOverrideConfig` from `runtime.models.text_encoder_overrides` after the loader seam carve (no behavior change).
- 2026-01-14: Flux engines now pass `expected_family` into `resolve_diffusion_bundle(...)` so prefixed Flux checkpoints can use metadata-driven signatures instead of state-dict detector guesses.
