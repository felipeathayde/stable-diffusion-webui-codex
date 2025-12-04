<!-- tags: frontend, utils, xyz -->
# apps/interface/src/utils Overview
Date: 2025-12-03
Owner: Frontend Maintainers
Last Review: 2025-12-03
Status: Active

## Purpose
- Small utility helpers shared across frontend modules (parsers, formatters, pure functions).

## Notes
- Keep helpers pure and framework-agnostic so they remain easy to unit test with Vitest.
- New utilities should ship with targeted tests under `apps/interface/src/**/`.
- 2025-12-03: Added XYZ helpers (`xyz.ts`) for axis parsing/combo building used by the sweep view/store.
