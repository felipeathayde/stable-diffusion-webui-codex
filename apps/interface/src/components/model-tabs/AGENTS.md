# apps/interface/src/components/model-tabs Overview
Date: 2026-03-13
Last Review: 2026-03-20
Status: Active

## Purpose
- Hold family-owned model-tab workspaces mounted by the canonical route owners under `src/views/**`.

## Files
- `apps/interface/src/components/model-tabs/WanVideoWorkspace.vue` — WAN txt2vid/img2vid workspace using `useVideoGeneration(tabId)`.
- `apps/interface/src/components/model-tabs/LtxVideoWorkspace.vue` — LTX video workspace using `useLtxVideoGeneration(tabId)` and the shared video-family presentation baseline.

## Notes
- Keep route ownership in `apps/interface/src/views/VideoModelTab.vue`.
- Do not invent a fake generic video composable or route-level workspace here; WAN and LTX remain family-owned.
- Shared presentational seams belong in `apps/interface/src/components/**`; shared styling belongs in `apps/interface/src/styles/**`.
- 2026-03-21: `WanVideoWorkspace.vue` now passes the WAN `4n+1` frame contract explicitly into the shared `VideoSettingsCard.vue`; the workspace owns WAN frame alignment instead of relying on shared-component defaults.
- 2026-03-20: `LtxVideoWorkspace.vue` now preserves raw manual LTX geometry/frame edits and leaves contract enforcement to the strict LTX lane (`32px` geometry, `8n+1` frames); width/height numeric entry stays unsnapped with the full `LTX_DIM_MAX` bound while slider/buttons still move in `32px` increments, and the shared `VideoSettingsCard.vue` remains reusable while family-owned frame rules stay in the workspace/store layer.
