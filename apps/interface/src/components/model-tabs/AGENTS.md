# apps/interface/src/components/model-tabs Overview
Date: 2026-03-13
Last Review: 2026-03-16
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
