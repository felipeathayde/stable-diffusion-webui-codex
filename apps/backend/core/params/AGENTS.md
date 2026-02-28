# apps/backend/core/params Overview
Date: 2025-10-28
Last Review: 2026-02-27
Status: Active

## Purpose
- Houses structured parameter translators that map high-level request payloads into engine/runtime-friendly dataclasses.

## Files
- `video.py` — Parameter loaders and validators for video-specific tasks (txt2vid/img2vid), including frame, resolution, and scheduling options.

## Notes
- When adding new task types, define parameter modules here to keep orchestration logic in `core` free from request-shape details.
- Ensure new schemas remain compatible with use-case orchestrators and are validated before reaching engines.
- 2026-02-27: `video.py` now includes `VideoUpscalingOptions` (SeedVR2 post-process contract) with strict typed fields mirrored by WAN router/stage validation.
