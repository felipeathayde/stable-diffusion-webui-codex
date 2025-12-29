# apps/backend/video Overview
Date: 2025-10-28
Owner: Video Runtime Maintainers
Last Review: 2025-12-16
Status: Active

## Purpose
- Houses shared video-specific helpers used across WAN22 and other video-capable pipelines.

## Subdirectories
- `interpolation/` — Video frame interpolation utilities (e.g., RIFE wrappers).
- `io/` — Input video probing/decoding (ffprobe/ffmpeg wrappers).
- `flow/` — Optical flow estimation + frame warping (torchvision RAFT).
- `export/` — Frame → video encoding (ffmpeg exporter; writes under `CODEX_ROOT/output`).

## Notes
- Keep video utilities generic so multiple engines/use cases can reuse them.
- Video IO/export requires `ffmpeg` + `ffprobe` on PATH; flow guidance requires `torch` + `torchvision`.
