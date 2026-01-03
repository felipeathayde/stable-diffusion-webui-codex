# apps/backend/debug — AGENTS
<!-- tags: backend, debug -->
Date: 2025-11-29
Owner: Backend Maintainers
Last Review: 2026-01-03
Status: Experimental

## Purpose
- Temporary debug utilities for side-by-side tracing and diagnosis.
- Safe to remove once parity investigations are complete; no runtime dependency for production paths unless explicitly enabled.

## Notes
- Do not import or execute code from `.refs/**` from `apps/**` (including debug helpers).
- 2026-01-02: Added standardized file header docstrings to debug modules (doc-only change; part of rollout).
- 2026-01-03: Added standardized file header docstring to `debug/__init__.py` (doc-only change; part of rollout).
