<!-- tags: frontend, utils, xyz -->
# apps/interface/src/utils Overview
Date: 2025-12-03
Owner: Frontend Maintainers
Last Review: 2026-01-29
Status: Active

## Purpose
- Small utility helpers shared across frontend modules (parsers, formatters, pure functions).

## Notes
- Keep helpers pure and framework-agnostic so they remain easy to unit test with Vitest.
- New utilities should ship with targeted tests under `apps/interface/src/**/`.
- 2025-12-03: Added XYZ helpers (`xyz.ts`) for axis parsing/combo building used by the sweep view/store.
- 2026-01-03: Added standardized file header block to `xyz.ts` (doc-only change; part of rollout).
- 2026-01-29: Added PNG infotext parsing + sampler/scheduler mapping helpers (`pnginfo.ts`) with unit tests.
