# apps/backend/runtime/processing Overview
Date: 2025-10-28
Last Review: 2026-03-22
Status: Active

## Purpose
- Shared preprocessing utilities (e.g., image conditioning, mask preparation) used before dispatching to engines.

## Notes
- 2026-03-22: `conditioners.py` now resolves a PyTorch-compatible img2img VAE-posterior seed from the processing object and forwards optional `encode_seed` into `encode_image_batch(...)`; this is the shared deterministic encode seam used by both unmasked and masked img2img prep.
- 2026-03-20: `conditioners.img2img_conditioning(...)` no longer synthesizes SD-family inpaint `c_concat` from checkpoint heuristics; shared conditioning stays request-owned and explicit.
- Centralize preprocessing logic here to avoid duplicating conversions in use cases or engines.
- `CodexProcessingBase` carries per-job smart flags (`smart_offload`, `smart_fallback`, `smart_cache`) so use-cases and engines can honor request-level overrides without consulting globals directly.
- 2026-01-02: Removed token-merging fields from processing dataclasses (feature is no longer supported).
- 2026-01-02: Added standardized file header docstrings to processing primitives (`__init__.py`, `conditioners.py`, `datatypes.py`) (doc-only change; part of rollout).
- 2026-01-29: `CodexProcessingImg2Img` now includes explicit mask enforcement selection (`post_blend` vs `per_step_clamp`) for Codex-native masked img2img.
- 2026-02-03: Processing models use `CodexHiresConfig` for hires configuration (renamed).
- 2026-02-08: `datatypes.py` now includes `ErSdeOptions` and `SamplingPlan.er_sde` to carry normalized ER-SDE runtime options through pipeline stages.
- 2026-02-08: `processing.models.RefinerConfig` now uses `swap_at_step` (serialized as `switch_at_step`) to represent swap-pointer semantics; `CodexHiresConfig.update_from_payload` reads nested refiner pointers from `refiner.switch_at_step`.
- 2026-02-10: Batch 4C tightened `datatypes.py` typing contracts with a closed init-image mode selector (`Literal["pixel","latent"]`), closed prompt controls (`_PromptControls`), and explicit `dict[str, object]` / `Mapping[str, object]` metadata surfaces for generation/video payload dataclasses.
- 2026-02-27: `conditioners.img2img_conditioning(...)` now masks-out the inpaint conditioning image (Forge/A1111 parity; default mask-weight behavior) before VAE encoding, reducing “conditioning sees removed content” drift for masked img2img.
- 2026-03-24: `CodexProcessingImg2Img` now carries `per_step_blend_strength` (`0..1`, default `1.0`) for masked `per_step_clamp`; this scalar is part of the typed runtime handoff, not an inferred hook-side default.
- 2026-03-25: `CodexProcessingTxt2Img.swap_model` is now a first-pass stage config (`SwapStageConfig`), while `CodexHiresConfig.swap_model` stays selector-only (`SwapModelConfig`). Do not collapse those two seams back into one type.
