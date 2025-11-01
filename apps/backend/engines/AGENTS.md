# apps/backend/engines Overview
Date: 2025-10-28
Owner: Engine Maintainers
Last Review: 2025-11-01
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
