<!-- tags: backend, types, payloads, samplers, exports -->

# apps/backend/types Overview
Date: 2026-01-03
Owner: Backend Maintainers
Last Review: 2026-01-03
Status: Active

## Purpose
- Holds lightweight backend type definitions and shared constants used across services, API schemas, and package facades.

## Key Files
- `__init__.py` — Re-export facade for common backend types/constants.
- `payloads.py` — Frozen key sets used for request payload validation.
- `samplers.py` — Canonical sampler enum + apply outcome container.
- `exports.py` — Lazy export name groups for backend `__getattr__` facades.

## Notes
- Keep these modules import-light to avoid bootstrap cycles; prefer simple dataclasses/enums and frozen sets.
- 2026-01-03: Added standardized file header docstrings to `types/*` modules (doc-only change; part of rollout).
