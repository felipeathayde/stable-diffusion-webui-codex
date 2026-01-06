<!-- tags: frontend, interface-src, overview -->
# apps/interface/src Overview
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2026-01-03
Status: Active

## Purpose
- Houses the Vue 3 application source code (components, state stores, API client, global styles).

## Subdirectories
- `api/` — Typed client wrappers and DTOs used to call the backend.
- `components/` — Reusable UI components.
- `stores/` — Pinia stores handling shared state.
- `styles/` — Scoped CSS modules imported via Tailwind tokens.
- `views/` — Page-level views mapped in the router.

## Key Files
- `App.vue` — Root component.
- `main.ts` — Application entrypoint (mounts Vue, installs plugins).
- `router.ts` — Route definitions for major views.
- `styles.css` — Global style entry (Tailwind + tokens).

## Notes
- Follow the frontend guidelines in `.sangoi/frontend/guidelines/` when adding new modules.
- Keep API types and schemas synchronized with `.sangoi/backend/interfaces/`.
- 2025-12-03: Vitest is available (`npm test`) for unit coverage; prompt serialization tests live under `components/prompt/`.
- 2025-12-04: Legacy `/txt2vid` and `/img2vid` SPA routes were removed; WAN22 video workflows now enter exclusively via model tabs (`/models/:tabId` with `type === 'wan'`) and backend video endpoints remain available for those tabs only.
- 2025-12-17: Added a shared `stores/workflows.ts` and guided-gen UI primitives (`styles/components/guided-gen.css`) to support WAN guided generation and reactive snapshots under `/workflows`.
- 2025-12-23: Added shared slider primitives under `components/ui/` (SliderField + NumberStepperInput) with matching styles under `styles/components/`.
- 2025-12-29: `App.vue` derives `--sticky-offset` from the `.main-header` height (via `ResizeObserver`) so `RunCard` can stay sticky below the header.
- 2026-01-01: Image model tabs now include `clipSkip` in their per-tab params and send `clip_skip`/`img2img_clip_skip` to the backend (no prompt-tag injection needed).
- 2026-01-03: Added standardized file header blocks to `App.vue`, `main.ts`, and `router.ts` (doc-only change; part of rollout).
