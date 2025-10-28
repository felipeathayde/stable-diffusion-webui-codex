# apps/backend/video Overview
Date: 2025-10-28
Owner: Video Runtime Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Houses shared video-specific helpers used across WAN22 and other video-capable pipelines.

## Subdirectories
- `interpolation/` — Video frame interpolation utilities (e.g., RIFE wrappers).

## Notes
- Keep video utilities generic so multiple engines/use cases can reuse them.
- Future video helpers (encoding, export) should live here rather than in engine-specific directories.
