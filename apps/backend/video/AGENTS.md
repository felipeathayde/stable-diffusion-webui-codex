# apps/backend/video Overview
Date: 2025-10-28
Last Review: 2026-02-27
Status: Active

## Purpose
- Houses shared video-specific helpers used across WAN22 and other video-capable pipelines.

## Subdirectories
- `interpolation/` — Video frame interpolation utilities (e.g., RIFE wrappers).
- `upscaling/` — SeedVR2 external CLI upscaling runner (repo-local runtime bootstrap + isolated venv; fail loud on missing deps/runtime requirements).
- `io/` — Input video probing/decoding (ffprobe/ffmpeg wrappers).
- `flow/` — Optical flow estimation + frame warping (torchvision RAFT).
- `export/` — Frame → video encoding (ffmpeg exporter; writes under `CODEX_ROOT/output`).

## Notes
- Keep video utilities generic so multiple engines/use cases can reuse them.
- Video IO/export resolves ffmpeg binaries via deterministic repo-local runtime paths first (`.uv/xdg-data/ffmpeg-downloader/ffmpeg`), then explicit env overrides/PATH.
- Default RIFE checkpoint is provisioned under `.uv/xdg-data/rife/rife47.pth`; interpolation runtime now attempts one-shot auto-provision only for default-token requests (custom paths/env overrides still fail loud).
- Flow guidance requires `torch` + `torchvision`.
- 2026-01-02: Added standardized file header docstrings to video export modules (doc-only change; part of rollout).
- 2026-02-23: RIFE/RAFT runtime defaults now derive device identity from memory-manager mount-device authority; unsupported device/backend combinations fail loud instead of implicit CUDA→CPU fallback.
- 2026-02-27: Added `upscaling/seedvr2_cli.py` to run SeedVR2 as an external CLI post-process stage (lossless ffmpeg intermediate, deterministic repo/model-dir resolution, strict frame count/size validation).
- 2026-02-27: SeedVR2 upscaling runtime default no longer couples to `.refs/**`; default repo bootstrap now targets `.uv/xdg-data/seedvr2/repo` (pinned ref) and CLI execution uses isolated `.uv/xdg-data/seedvr2/venv` dependency provisioning.
