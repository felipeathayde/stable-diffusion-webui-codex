# apps/backend/runtime/families/sd Overview
Date: 2025-10-28
Last Review: 2026-02-01
Status: Active

## Purpose
- Stable Diffusion (SD) runtime helpers used by SD engines (conditioning, pipelines, control modules).

## Subdirectories
- `cnets/` — ControlNet-specific helpers and wrappers.

## Notes
- Keep SD runtime modules aligned with `apps/backend/engines/sd/`.
- 2026-01-02: Added standardized file header docstrings to `__init__.py` (doc-only change; part of rollout).
- 2026-02-01: Added `hires_fix.py` (hires pass init preparation; routes latent vs spandrel upscalers via the global upscalers runtime).
