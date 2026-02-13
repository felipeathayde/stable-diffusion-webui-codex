# apps/backend/video/interpolation Overview
Date: 2025-10-28
Last Review: 2026-02-13
Status: Active

## Purpose
- Provides video frame interpolation helpers (currently RIFE) shared by txt2vid/img2vid pipelines.

## Key Files
- `rife.py` — In-repo RIFE adapter via `ccvfi`; resolves deterministic model paths and raises explicit errors when runtime/model assets are missing.

## Notes
- Keep interpolation helpers lightweight and stateless. Interpolation runtime errors must fail loud (no silent fallback remapping).
- `rife47.pth` token resolves to repo-local runtime storage (`.uv/xdg-data/rife/rife47.pth`) and must fail loud when missing.
- 2026-02-13: `rife.py` now applies explicit runtime cleanup (`unload` hook + CUDA cache cleanup + GC) in a `finally` path so lifecycle is deterministic on both success and failure.
- 2026-01-03: Added standardized file header docstrings to interpolation modules (doc-only change; part of rollout).
