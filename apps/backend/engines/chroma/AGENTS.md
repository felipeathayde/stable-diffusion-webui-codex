# apps/backend/engines/chroma Overview
Date: 2025-10-28
Owner: Engine Maintainers
Last Review: 2026-01-02
Status: Active

## Purpose
- Chroma engine implementations leveraging the Chroma runtime helpers.

## Notes
- Align runtime and engine changes; move shared pieces into `runtime/chroma/` when appropriate.
- Chroma reuses the Flux engine toolkit (`apps/backend/engines/flux/spec.py`); extend specs and rely on `_build_components` to assemble runtime state during `load()`.
- Engine capabilities now advertise txt2img/img2img support and enforce runtime guards before accessing Flux structures.
- 2026-01-02: Added standardized file header docstrings to Chroma engine modules (doc-only change; part of rollout).
