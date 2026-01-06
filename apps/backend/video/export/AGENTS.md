# apps/backend/video/export Overview
Date: 2025-12-16
Owner: Video Runtime Maintainers
Last Review: 2026-01-03
Status: Active

## Purpose
- Encode frame sequences to a video container (mp4/webm/gif) using ffmpeg.

## Key files
- `apps/backend/video/export/ffmpeg_exporter.py` — `export_video()` writes frame PNGs to a temp dir then runs ffmpeg.

## Notes
- Output root is `CODEX_ROOT/output` (repo-local) and served via `/api/output/{rel_path}`.
- Backend must serve outputs via a root-scoped file route (see `/api/output/{rel_path}`) rather than exposing arbitrary paths.
- Export errors should be explicit (`VideoExportError`) so users can fix missing ffmpeg/codec issues.
- 2026-01-02: Added standardized file header docstrings to exporter modules (doc-only change; part of rollout).
- 2026-01-03: Added standardized file header docstring to `export/__init__.py` (doc-only change; part of rollout).
