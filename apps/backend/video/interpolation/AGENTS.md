# apps/backend/video/interpolation Overview
Date: 2025-10-28
Owner: Video Runtime Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Provides video frame interpolation helpers (currently RIFE) shared by txt2vid/img2vid pipelines.

## Key Files
- `rife.py` — Optional RIFE integration; raises explicit errors when model assets are missing or interpolation is disabled.

## Notes
- Keep interpolation helpers lightweight and stateless. Engines should supply context (frames, options) and handle fallbacks when interpolation is unavailable.
