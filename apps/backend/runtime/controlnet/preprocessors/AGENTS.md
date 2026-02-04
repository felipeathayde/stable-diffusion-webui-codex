# AGENT — apps/backend/runtime/controlnet/preprocessors
Date: 2025-10-31
Last Review: 2026-01-20
Status: Active

## Purpose
- Host Codex-native preprocessing pipelines for ControlNet (edge detectors, depth estimators, pose extractors, etc.).
- Provide a registry (`ControlPreprocessorRegistry`) for runtime/extension consumers to resolve preprocessors by slug.
- Implement differentiable and deterministic transformations without importing legacy modules.

## Notes
- Model implementations live under `models/` (HED, PiDiNet, MLSD, lineart_anime, manga_line, LeReS, Zoe, etc.). Public preprocessor entrypoints and registration wiring are still pending.
- `__init__.py` is a package marker (no auto-registration at import time); consumers should import `ControlPreprocessorRegistry` from `registry.py` and explicitly register any preprocessors they expose.
- Future batches (pose, segmentation, geometry) should register under the same registry and update the parity matrix (`.sangoi/backend/runtime/controlnet-parity.md`).
