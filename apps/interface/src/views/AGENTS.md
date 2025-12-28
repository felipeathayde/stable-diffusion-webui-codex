# apps/interface/src/views Overview
<!-- tags: frontend, views, model-tabs -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-28
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
- `ImageModelTab.vue` mirrors the legacy engine-tab layout (same `panels` + `panel-stack` structure as WAN): PromptCard (progress/error + optional init-image controls), BasicParametersCard + optional Highres/Refiner, RunCard (batch dropdown), ResultsCard (gentime/actions), ResultViewer gallery, and an Info panel; generation/payload wiring lives in `useGeneration(tabId)` and capability gating uses `useEngineCapabilitiesStore()`.
- 2025-12-28: Replaced the remaining small switch widgets (`qs-switch--sm`) in `Home.vue` and `WANTab.vue` with `qs-toggle-btn` buttons for consistent toggle styling across the UI.
- 2025-12-28: `WANTab.vue` now uses `WanSubHeader` for consistent section headers and merges Video + Video Output into a single card above High/Low Noise.
