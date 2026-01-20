# apps/backend/runtime/vision/clip Overview
Date: 2025-10-31
Owner: Backend Runtime Maintainers
Last Review: 2026-01-18
Status: Active

## Purpose
- Codex-native Clip vision runtime (model specs, state-dict tooling, encoder wrapper, preprocessing).

## Notes
- Raise `ClipVisionError` subclasses for all error paths; never fall back to silent prints.
- Specs live in `specs.py` and feed detection/registry helpers; extend via dataclasses/enums.
- `ClipVisionEncoder` handles device/dtype selection and logging; keep it free of UI concerns.
- Update `.sangoi/plans/codex-legacy-backlog.md` entry #9 when behaviour changes or new variants land.
- 2026-01-02: Added standardized file header docstrings to CLIP vision runtime modules (doc-only change; part of rollout).
- 2026-01-18: `clip/__init__.py` is a package marker (no re-exports); import types/helpers from the defining modules (`encoder.py`, `errors.py`, `types.py`, etc.).
