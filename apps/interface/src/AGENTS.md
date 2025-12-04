<!-- tags: frontend, interface-src, overview -->
# apps/interface/src Overview
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-03
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
