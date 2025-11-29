# apps/backend/debug — AGENTS
<!-- tags: backend, debug, legacy-mimic -->
Date: 2025-11-29
Owner: Backend Maintainers
Status: Experimental

## Purpose
- Temporary debug utilities to mirror the legacy (.legacy) pipeline for side-by-side tracing and diagnosis.
- Safe to remove once Codex pipelines reach parity; no runtime dependency for production paths unless explicitly enabled.

## Notes
- Imports from `.legacy` are permitted here **only** for debugging. Do not propagate `.legacy` imports elsewhere.
- Call `apps.backend.debug.legacy_mimic.enable()` to install hooks; otherwise no behaviour changes occur.
