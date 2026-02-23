# apps/backend/video/flow Overview
Date: 2025-12-16
Last Review: 2026-02-23
Status: Active

## Purpose
- Provide optical flow guidance for video-to-video workflows (estimate flow + warp prior outputs).

## Key files
- `apps/backend/video/flow/torchvision_raft.py` — RAFT (torchvision) estimator + `warp_frame()` (lazy-loaded).

## Notes
- Imports are intentionally lazy so environments without torch/torchvision can still import the backend.
- Fail fast when flow guidance is enabled but dependencies are missing (`FlowGuidanceError`).
- Warping operates on PIL images; tensors are transient and should stay on the chosen device.
- 2026-01-18: `flow/__init__.py` is now a package marker (no re-exports); import flow helpers from `apps/backend/video/flow/torchvision_raft.py`.
- 2026-02-23: `torchvision_raft.py` removed hardcoded fallback device literals (`"cuda"` / `"cpu"`); default RAFT/warp device now resolves via memory-manager mount authority when caller/flow tensors do not provide a device.
