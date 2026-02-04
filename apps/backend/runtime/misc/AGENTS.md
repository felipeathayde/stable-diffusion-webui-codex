# apps/backend/runtime/misc Overview
Date: 2025-10-28
Last Review: 2026-01-20
Status: Active

## Purpose
- Miscellaneous runtime helpers (logging, strict checks, utility functions) that support other runtime modules.

## Notes
- Periodically review this directory; migrate modules into more specific packages when patterns emerge.
- 2026-01-02: Added standardized file header docstrings to `__init__.py`, `checkpoint_pickle.py`, `diffusers_state_dict.py`, and `image_resize.py` (doc-only change; part of rollout).
- 2026-01-18: `misc/__init__.py` is now a package marker (no re-exports); import helpers from their defining modules (e.g. `image_resize.py`, `sub_quadratic_attention.py`).
- 2026-01-20: Removed unused `tomesd.py` helper (no call sites).
