# apps/interface/src/stores Overview
<!-- tags: frontend, stores, state -->
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
- 2025-12-03: Stores track two refiner configs: a global `refiner` (for post-base pass) and `highres.refiner` nested under the hires options for a hires-coupled refiner stage.
- 2025-12-03: `xyz.ts` store runs frontend-driven XYZ sweeps (X/Y/Z axes) using the current SDXL form as baseline, with stop support and per-cell status.
- 2025-12-03: XYZ store now enqueues payload snapshots, supports stop-after-current vs stop-now (calling `/api/tasks/{id}/cancel`), and preserves hires/refiner in each job payload.
