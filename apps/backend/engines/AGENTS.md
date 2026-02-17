# apps/backend/engines Overview
<!-- tags: backend, engines, registry, lazy-imports -->
Date: 2025-12-05
Last Review: 2026-02-17
Status: Active

## Purpose
- Implements model-specific execution logic (Diffusion engines, WAN22, Flux, SD, Chroma), plus shared engine utilities and registration.

## Subdirectories
- `common/` — Shared base classes/helpers used by multiple engines.
- `sd/`, `flux/`, `wan22/`, `zimage/`, `anima/` — Model-specific engine implementations and components (Flux family includes Chroma + Kontext variants).
- `util/` — Utility helpers for schedulers, attention mapping, etc.

## Key Files
- `__init__.py` — Exposes engine registration hooks and lazy engine-class exports (import-light public surface for external consumers).
- `registration.py` — Canonical registry of available engines.

## Notes
- New engines should live under their own subdirectory (mirroring existing patterns) and register via `registration.py`.
- Task orchestration lives under `apps/backend/use_cases/`; keep engines model-specific and keep shared logic under `apps/backend/runtime/`.
- Factory standard: for each model family, prefer a `spec.py` + `factory.py` seam under `apps/backend/engines/<family>/` so assembly stays consistent and engines keep only model-specific logic (plan: `.sangoi/plans/2026-01-03-engine-factory-standard-v1.md`).
- Diffusion engines now share a `BaseInferenceEngine` lifecycle; instantiate via the registry and let `load()` pull `DiffusionModelBundle` instances instead of invoking loaders manually.
- 2025-11-03: SDXL engine reads `debug_conditioning` from backend config (no direct env lookup) to log conditioning norms when requested.
- 2025-11-30: `apps.backend.engines.__init__` now lazily resolves WAN22 engine classes; importing the package no longer pulls Hugging Face assets or torch unless the engines are requested.
- 2025-12-05: `common.base.BaseInferenceEngine.load()` now accepts an optional `text_encoder_override` option (family + `<family>/<path>` label from paths.json [+ optional components]) and forwards it to `runtime.models.resolve_diffusion_bundle`, so text encoder overrides are applied centrally by the loader instead of por-engine shims.
- 2026-01-02: Added standardized file header docstrings to engine facade/registration modules (doc-only change; part of rollout).
- 2026-01-04: Flux family engine keys are `flux1` / `flux1_kontext` / `flux1_chroma` (no legacy aliases) to make room for a future Flux.2.
- 2026-01-06: `common.base` VAE overrides (`vae_path`) now unwrap wrapper VAEs via `first_stage_model` before applying state dicts.
- 2026-01-18: `register_default_engines(...)` now registers `flux1_chroma` alongside `flux1`/`flux1_kontext` so the canonical Chroma engine key is available to API callers without manual registration.
- 2026-01-31: Image-mode wrappers are now fully owned by `apps/backend/use_cases/` (txt2img + img2img); engines delegate via `CodexDiffusionEngine`. The common base also provides default first-stage VAE encode/decode for image engines (WAN/video keep explicit overrides).
- 2026-01-31: Engine-common helpers expanded to reduce drift:
  - Generic conditioning cache helpers (overrideable per call) + shared tensor move helpers for CPU↔device caching.
  - Runtime guard helper (`require_runtime`) for consistent “call load() first” errors across engines.
- 2026-02-05: Added the `anima` engine key and a stub `AnimaEngine` facade (fail-loud until runtime is ported).
- 2026-02-08: Engine adapters now map swap-model pointer semantics using `switch_at_step` (`RefinerConfig.swap_at_step`) for both global and hires nested refiner config.
- 2026-02-17: WAN22 canonical registration now maps `wan22_14b` to a dedicated GGUF 14B lane (`Wan2214BGgufEngine`) with no inheritance from `wan22_5b`, preventing 14B dispatch from collapsing into 5B behavior.
