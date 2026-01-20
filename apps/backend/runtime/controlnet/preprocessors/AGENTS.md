# AGENT — apps/backend/runtime/controlnet/preprocessors
Date: 2025-10-31
Owner: Backend Runtime Maintainers
Last Review: 2026-01-18
Status: Active

## Purpose
- Host Codex-native preprocessing pipelines for ControlNet (edge detectors, depth estimators, pose extractors, etc.).
- Provide a registry (`ControlPreprocessorRegistry`) for runtime/extension consumers to resolve preprocessors by slug.
- Implement differentiable and deterministic transformations without importing legacy modules.

## Notes
- Initial batch covers edge detectors (`canny`, `binary`, `sobel`, `lineart`, `hed`, `pidinet`, `mlsd`, `lineart_anime`, `manga_line`) and depth (`depth_dpt_hybrid`, `depth_leres`, `depth_zoe`); neural models load quality weights from `~/.cache/codex/controlnet/`, and canny falls back to a manual torch implementation when `torchvision` is unavailable.
- All preprocessors accept tensors shaped `[B, C, H, W]` (float) and return `PreprocessorResult` with metadata describing thresholds/parameters.
- `__init__.py` is a package marker (no auto-registration at import time); callers should import `ControlPreprocessorRegistry` from `registry.py` and register built-ins explicitly via `edges.py` / `depth.py`.
- Future batches (depth, pose, segmentation) should register under the same registry and update the parity matrix (`.sangoi/backend/runtime/controlnet-parity.md`).
