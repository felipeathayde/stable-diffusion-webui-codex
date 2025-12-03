# apps/interface/src/stores Overview
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-03
Status: Active

## Purpose
- Pinia stores encapsulating shared UI/application state (engine selections, task options, session data).

## Notes
- Keep store interfaces aligned with backend schemas and avoid duplicating validation already enforced server-side.
- Derive computed state for components instead of mutating raw backend payloads.
- `sdxl.ts` mirrors `txt2img.ts` with Codex SDXL defaults; prefer reusing shared helpers when adjusting both stores.
- 2025-11-03: SDXL store persists profiles locally (`loadProfile`/`saveProfile`) so the `/sdxl` view can reuse saved parameters.
- 2025-12-03: Result `info` now includes prompt, negative prompt, resolved seed, and default save directory so the UI surfaces real generation inputs/outputs.
