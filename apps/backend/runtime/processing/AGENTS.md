# apps/backend/runtime/processing Overview
Date: 2025-10-28
Last Review: 2026-02-08
Status: Active

## Purpose
- Shared preprocessing utilities (e.g., image conditioning, mask preparation) used before dispatching to engines.

## Notes
- Centralize preprocessing logic here to avoid duplicating conversions in use cases or engines.
- `CodexProcessingBase` carries per-job smart flags (`smart_offload`, `smart_fallback`, `smart_cache`) so use-cases and engines can honor request-level overrides without consulting globals directly.
- 2026-01-02: Removed token-merging fields from processing dataclasses (feature is no longer supported).
- 2026-01-02: Added standardized file header docstrings to processing primitives (`__init__.py`, `conditioners.py`, `datatypes.py`) (doc-only change; part of rollout).
- 2026-01-29: `CodexProcessingImg2Img` now includes explicit mask enforcement selection (`post_blend` vs `per_step_clamp`) for Codex-native masked img2img.
- 2026-02-03: Processing models use `CodexHiresConfig` for hires configuration (renamed).
- 2026-02-08: `datatypes.py` now includes `ErSdeOptions` and `SamplingPlan.er_sde` to carry normalized ER-SDE runtime options through pipeline stages.
- 2026-02-08: `processing.models.RefinerConfig` now uses `swap_at_step` (serialized as `switch_at_step`) to represent swap-pointer semantics; `CodexHiresConfig.update_from_payload` reads nested refiner pointers from `refiner.switch_at_step`.
