# apps/backend/runtime/vision Overview
Date: 2025-10-31
Last Review: 2026-02-10
Status: Active

## Purpose
- Vision-specific runtime modules that complement backend engines (CLIP vision encoders, SigLIP, DINO, etc.).

## Notes
- Keep implementations dataclass-driven with explicit validation and logging.
- No fallback behaviour: raise descriptive `ClipVisionError` subclasses on unsupported configs.
- When adding new encoder families, document invariants here and link to the relevant plan entry under `.sangoi/plans/`.
- 2026-01-02: Added standardized file header docstrings to vision runtime modules (doc-only change; part of rollout).
- 2026-01-18: `vision/__init__.py` is a package marker (no re-exports); import subpackages/modules explicitly (e.g. `vision/clip/*`).
- 2026-02-10: Structural conversion operations in vision state-dict converters follow global policy `CODEX_WEIGHT_STRUCTURAL_CONVERSION` (`auto` fail-loud / `convert` opt-in).
