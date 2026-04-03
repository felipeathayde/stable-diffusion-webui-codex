# apps/interface/src/components/model-tabs Overview
Date: 2026-03-13
Last Review: 2026-04-03
Status: Active

## Purpose
- Hold frozen reference video bodies used by the canonical route owners under `src/views/**`.

## Files
- `apps/interface/src/components/model-tabs/WanVideoWorkspace.vue` — frozen source video workspace kept as the mechanical reference for the `VideoModelTab.vue` cutover.

## Notes
- `apps/interface/src/views/VideoModelTab.vue` is now the canonical baseline video workspace under `src/views/**`.
- `apps/interface/src/views/VideoTabRouteView.vue` is the thin route selector while current video families still branch.
- `apps/interface/src/views/video-model/**` now owns the live family-specific runtime helpers (`VideoModelTabWanRuntime.vue`, `VideoModelTabLtxRuntime.vue`).
- Keep `WanVideoWorkspace.vue` frozen as the mechanical source/reference until the follow-on convergence tranche lands.
- Do not invent a second generic video owner or compatibility layer here.
- Shared presentational seams belong in `apps/interface/src/components/**`; shared styling belongs in `apps/interface/src/styles/**`.
- 2026-03-21: `WanVideoWorkspace.vue` now passes the WAN `4n+1` frame contract explicitly into the shared `VideoSettingsCard.vue`; the workspace owns WAN frame alignment instead of relying on shared-component defaults.
- 2026-04-03: LTX no longer has a live body owner in this folder; the strict geometry/frame/execution-profile contract now renders through `apps/interface/src/views/VideoModelTab.vue` plus `apps/interface/src/views/video-model/VideoModelTabLtxRuntime.vue`.
