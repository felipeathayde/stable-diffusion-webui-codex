# apps/interface/src/views/video-model Overview
Date: 2026-04-03
Last Review: 2026-04-03
Status: Active

## Purpose
- Hold view-local runtime helpers for `VideoModelTab.vue`.
- These helpers mount family-specific composables/watchers and expose slot props to the route-owned video view.

## Files
- `apps/interface/src/views/video-model/VideoModelTabWanRuntime.vue` — renderless WAN runtime helper for the canonical video tab view.
- `apps/interface/src/views/video-model/VideoModelTabLtxRuntime.vue` — renderless LTX runtime helper for the canonical video tab view.

## Notes
- `VideoModelTab.vue` remains the body/layout owner. Do not move panel/card template ownership into this folder.
- Helpers here may own active-family-only side effects (bootstrap, auto-resume, temporal persistence, guided listeners, checkpoint-default watchers).
- Do not add shared presentational components here; shared UI belongs under `apps/interface/src/components/**`.
- Only the active video family may instantiate its runtime helper at a time.
