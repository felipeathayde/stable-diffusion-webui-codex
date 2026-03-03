# apps/interface/tools Overview
Date: 2025-10-28
Last Review: 2026-03-03
Status: Active

## Purpose
- Contains scripts/utilities that support frontend development (e.g., port guard, lint/typecheck helpers).

## Key Files
- `port-guard-dev.mjs` — Dev server launcher wrapper that enforces safe/available UI ports.
- `ui-consistency-report.mjs` — UI style-contract scanner (inline + dynamic + scoped + selector duplication + docs/toolchain drift), with strict mode (`--strict`) for fail-loud gating.

## Notes
- Keep these scripts in sync with repository-shipped frontend docs (`apps/interface/AGENTS.md`, `apps/interface/src/styles/AGENTS.md`) so developers know how to invoke them.
- 2026-02-21: Added `ui-consistency-report.mjs`; `npm run verify` now runs strict style-contract gating (`report:ui-consistency:strict`) before typecheck/build.
- 2026-02-21: Dynamic style bindings are now scanned; strict mode fails on disallowed dynamic `:style` usage while docs/toolchain drift remains report-only.
- 2026-03-03: `ui-consistency-report.mjs` docs/toolchain drift references were moved to repo-shipped docs paths under `apps/interface/**` (no `.sangoi` dependency).
