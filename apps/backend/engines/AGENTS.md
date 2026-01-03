# apps/backend/engines Overview
<!-- tags: backend, engines, registry, lazy-imports -->
Date: 2025-12-05
Owner: Engine Maintainers
Last Review: 2026-01-02
Status: Active

## Purpose
- Implements model-specific execution logic (Diffusion engines, WAN22, Flux, SD, Chroma), plus shared engine utilities and registration.

## Subdirectories
- `common/` — Shared base classes/helpers used by multiple engines.
- `diffusion/` — Task-layer diffusion runners (txt2img, img2img, txt2vid, img2vid) that wire engines into use cases.
- `sd/`, `flux/`, `chroma/`, `wan22/` — Model-specific engine implementations and components.
- `util/` — Utility helpers for schedulers, attention mapping, etc.

## Key Files
- `__init__.py` — Exposes engine registration hooks.
- `registration.py` — Canonical registry of available engines.

## Notes
- New engines should live under their own subdirectory (mirroring existing patterns) and register via `registration.py`.
- Shared diffusion logic belongs in `diffusion/`; avoid duplicating orchestration that already exists in `use_cases/`.
- Diffusion engines now share a `BaseInferenceEngine` lifecycle; instantiate via the registry and let `load()` pull `DiffusionModelBundle` instances instead of invoking loaders manually.
- 2025-11-03: SDXL engine reads `debug_conditioning` from backend config (no direct env lookup) to log conditioning norms when requested.
- 2025-11-30: `apps.backend.engines.__init__` now lazily resolves WAN22 engine classes; importing the package no longer pulls Hugging Face assets or torch unless the engines are requested.
- 2025-12-05: `common.base.BaseInferenceEngine.load()` now accepts an optional `text_encoder_override` option (family + `/api/text-encoders` label [+ optional components]) and forwards it to `runtime.models.resolve_diffusion_bundle`, so text encoder overrides are applied centrally by the loader instead of por-engine shims.
- 2026-01-02: Added standardized file header docstrings to engine facade/registration modules (doc-only change; part of rollout).
