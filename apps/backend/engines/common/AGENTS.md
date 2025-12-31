# apps/backend/engines/common Overview
Date: 2025-10-28
Owner: Engine Maintainers
Last Review: 2025-12-30
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
