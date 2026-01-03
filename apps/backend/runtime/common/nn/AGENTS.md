# apps/backend/runtime/common/nn Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2026-01-02
Status: Active

## Purpose
- Shared neural network building blocks (layers, wrappers) used across multiple runtimes/engines.

## Notes
- Keep modules generic; specialize behaviour in the model-specific runtimes instead of here.
- 2026-01-02: Added standardized file header docstrings to `__init__.py`, `base.py`, `clip.py`, `t5.py`, `clip_text_cx.py`, and `unet/{__init__,config,utils}.py` (doc-only change; part of rollout).
