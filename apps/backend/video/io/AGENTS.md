# apps/backend/video/io Overview
Date: 2025-12-16
Owner: Video Runtime Maintainers
Last Review: 2026-01-03
Status: Active

## Purpose
- Provide ffmpeg/ffprobe-backed video probing and frame extraction utilities for backend video tasks.

## Key files
- `apps/backend/video/io/ffmpeg.py` — `probe_video()` and `extract_frames()` wrappers (fail-fast, explicit errors).

## Notes
- Keep imports minimal and use subprocess calls (no cv2 dependency).
- All output paths must be explicit and sandbox-safe (callers should write under `./tmp/`).
- Raise `FFmpegUnavailableError` when ffmpeg/ffprobe are missing instead of silently degrading.
- 2026-01-02: Added standardized file header docstrings to video IO modules (doc-only change; part of rollout).
- 2026-01-03: Added standardized file header docstring to `io/__init__.py` (doc-only change; part of rollout).
