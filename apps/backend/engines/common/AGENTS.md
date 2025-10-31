# apps/backend/engines/common Overview
Date: 2025-10-28
Owner: Engine Maintainers
Last Review: 2025-10-31
Status: Active

## Purpose
- Shared engine utilities (base classes, mixins, helpers) reused across SD, Flux, Chroma, WAN22 engines.

## Notes
- `CodexDiffusionEngine` now owns a component tracker (`bind_components`) that keeps active/original/LoRA snapshots; use the setter instead of manually cloning.
- Model family flags (`is_sd1`, `is_sd2`, `is_sd3`, `is_sdxl`) are exposed as read-only properties; call `register_model_family(...)` during engine initialisation.
- Tiling/CFG scale toggles remain available but emit structured logs when changed.
