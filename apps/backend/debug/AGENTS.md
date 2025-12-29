# apps/backend/debug — AGENTS
<!-- tags: backend, debug, legacy-mimic -->
Date: 2025-11-29
Owner: Backend Maintainers
Last Review: 2025-12-29
Status: Experimental

## Purpose
- Temporary debug utilities to mirror the legacy Forge pipeline (snapshot under `.refs/Forge-A1111`) for side-by-side tracing and diagnosis.
- Safe to remove once Codex pipelines reach parity; no runtime dependency for production paths unless explicitly enabled.

## Notes
- Imports from `.refs/Forge-A1111` are permitted here **only** for debugging. Do not propagate `.refs` imports elsewhere.
- Call `apps.backend.debug.legacy_mimic.enable()` to install hooks; otherwise no behaviour changes occur.
- 2025-12-29: Debug helpers now anchor `.refs/*` lookups under `CODEX_ROOT` (required) so they don’t depend on the process CWD.
