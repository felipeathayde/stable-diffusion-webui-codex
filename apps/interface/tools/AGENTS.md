# apps/interface/tools Overview
Date: 2025-10-28
Last Review: 2026-02-21
Status: Active

## Purpose
- Contains scripts/utilities that support frontend development (e.g., port guard, lint/typecheck helpers).

## Key Files
- `port-guard-dev.mjs` — Dev server launcher wrapper that enforces safe/available UI ports.
- `ui-consistency-report.mjs` — Report-only scanner for frontend style-contract drift (inline/scoped styles, selector duplication, docs/toolchain drift).

## Notes
- Keep these scripts in sync with documentation in `.sangoi/frontend/` so developers know how to invoke them.
- 2026-02-21: Added `ui-consistency-report.mjs`; `npm run verify` now runs report generation before typecheck/test/build (report-only, non-blocking).
