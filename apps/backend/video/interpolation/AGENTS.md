# apps/backend/video/interpolation Overview
Date: 2025-10-28
Owner: Video Runtime Maintainers
Last Review: 2026-01-03
Status: Active

## Purpose
- Provides video frame interpolation helpers (currently RIFE) shared by txt2vid/img2vid pipelines.

## Key Files
- `rife.py` — Optional RIFE integration; raises explicit errors when model assets are missing or interpolation is disabled.

## Notes
- Keep interpolation helpers lightweight and stateless. Engines should supply context (frames, options) and handle fallbacks when interpolation is unavailable.
- 2026-01-03: Added standardized file header docstrings to interpolation modules (doc-only change; part of rollout).
