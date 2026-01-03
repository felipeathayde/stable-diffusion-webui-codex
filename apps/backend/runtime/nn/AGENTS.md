# apps/backend/runtime/nn Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2026-01-02
Status: Active

## Purpose
- Model-agnostic neural network helpers (e.g., shared modules, utilities) plugged into runtime pipelines.

## Notes
- Keep this package minimal; use model-specific folders (`sd`, `wan22`, etc.) for specialized logic.
- 2026-01-02: Added standardized file header docstrings to the `__init__.py` re-export facade (doc-only change; part of rollout).
