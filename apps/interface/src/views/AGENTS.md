# apps/interface/src/views Overview
<!-- tags: frontend, views, model-tabs -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-27
Status: Active

## Purpose
- Page-level Vue components mapped to routes (e.g., generation workspace, settings).

## Notes
- Views should compose reusable components and stores; avoid duplicating logic that belongs in shared modules.
- Keep routes documented in `apps/interface/src/router.ts` and the UI taxonomy in `.sangoi/frontend/guidelines/`.
- All generation workspaces live under model tabs (`/models/:tabId`):
  - `ModelTabView.vue` mounts `WANTab.vue` when `tab.type === 'wan'`.
  - `ModelTabView.vue` mounts `ImageModelTab.vue` when `tab.type` is `sd15|sdxl|flux|zimage`.
- `Home.vue` is the engine-agnostic landing page and the canonical place to manage tabs (enable/disable, rename, duplicate, remove).
- `WANTab.vue` uses typed WAN video payload builders and `useVideoGeneration(tabId)` for streaming progress.
- `ImageModelTab.vue` uses `useGeneration(tabId)` for txt2img/img2img and renders shared Results/Run cards (`components/results/*`).
