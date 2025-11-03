# apps/interface/src/stores Overview
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-11-03
Status: Active

## Purpose
- Pinia stores encapsulating shared UI/application state (engine selections, task options, session data).

## Notes
- Keep store interfaces aligned with backend schemas and avoid duplicating validation already enforced server-side.
- Derive computed state for components instead of mutating raw backend payloads.
- `sdxl.ts` mirrors `txt2img.ts` with Codex SDXL defaults; prefer reusing shared helpers when adjusting both stores.
- 2025-11-03: SDXL store persists profiles locally (`loadProfile`/`saveProfile`) so the `/sdxl` view can reuse saved parameters.
