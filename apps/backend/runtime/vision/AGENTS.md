# apps/backend/runtime/vision Overview
Date: 2025-10-31
Owner: Backend Runtime Maintainers
Last Review: 2025-10-31
Status: Active

## Purpose
- Vision-specific runtime modules that complement backend engines (CLIP vision encoders, SigLIP, DINO, etc.).

## Notes
- Keep implementations dataclass-driven with explicit validation and logging.
- No fallback behaviour: raise descriptive `ClipVisionError` subclasses on unsupported configs.
- When adding new encoder families, document invariants here and link to the relevant plan entry under `.sangoi/plans/`.
