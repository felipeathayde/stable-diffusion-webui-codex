# apps/interface/src/components/model-tabs Overview
Date: 2026-03-13
Last Review: 2026-04-03
Status: Active

## Purpose
- Hold supporting video workspaces and frozen reference bodies used by the canonical route owners under `src/views/**`.

## Files
- `apps/interface/src/components/model-tabs/WanVideoWorkspace.vue` — frozen source video workspace kept as the mechanical reference for the `VideoModelTab.vue` cutover.
- `apps/interface/src/components/model-tabs/LtxVideoWorkspace.vue` — live LTX video workspace using `useLtxVideoGeneration(tabId)` and the shared video presentation structure.

## Notes
- `apps/interface/src/views/VideoModelTab.vue` is now the canonical baseline video workspace under `src/views/**`.
- `apps/interface/src/views/VideoTabRouteView.vue` is the thin route selector while current video families still branch.
- Keep `WanVideoWorkspace.vue` frozen as the mechanical source/reference until the follow-on convergence tranche lands.
- Do not invent a second generic video owner or compatibility layer here.
- Shared presentational seams belong in `apps/interface/src/components/**`; shared styling belongs in `apps/interface/src/styles/**`.
- 2026-03-21: `WanVideoWorkspace.vue` now passes the WAN `4n+1` frame contract explicitly into the shared `VideoSettingsCard.vue`; the workspace owns WAN frame alignment instead of relying on shared-component defaults.
- 2026-03-20: `LtxVideoWorkspace.vue` now preserves raw manual LTX geometry/frame edits and leaves contract enforcement to the strict LTX lane (`32px` geometry, `8n+1` frames); width/height numeric entry stays unsnapped with the full `LTX_DIM_MAX` bound while slider/buttons still move in `32px` increments, and the shared `VideoSettingsCard.vue` remains reusable while family-owned frame rules stay in the workspace/store layer.
