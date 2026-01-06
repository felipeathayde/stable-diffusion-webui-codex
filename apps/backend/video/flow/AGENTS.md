# apps/backend/video/flow Overview
Date: 2025-12-16
Owner: Video Runtime Maintainers
Last Review: 2026-01-03
Status: Active

## Purpose
- Provide optical flow guidance for video-to-video workflows (estimate flow + warp prior outputs).

## Key files
- `apps/backend/video/flow/torchvision_raft.py` — RAFT (torchvision) estimator + `warp_frame()` (lazy-loaded).

## Notes
- Imports are intentionally lazy so environments without torch/torchvision can still import the backend.
- Fail fast when flow guidance is enabled but dependencies are missing (`FlowGuidanceError`).
- Warping operates on PIL images; tensors are transient and should stay on the chosen device.
- 2026-01-03: Added standardized file header docstrings to `flow/__init__.py` and `flow/torchvision_raft.py` (doc-only change; part of rollout).
