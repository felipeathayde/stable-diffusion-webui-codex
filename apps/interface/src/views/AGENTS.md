# apps/interface/src/views Overview
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Page-level Vue components mapped to routes (e.g., generation workspace, settings).

## Notes
- Views should compose reusable components and stores; avoid duplicating logic that belongs in shared modules.
- Keep routes documented in `router.ts` and the UI taxonomy in `.sangoi/frontend/guidelines/`.
