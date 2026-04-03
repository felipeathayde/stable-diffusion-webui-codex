# apps/interface/src/components/video Overview
Date: 2026-04-03
Last Review: 2026-04-03
Status: Active

## Purpose
- Shared presentational video card components used by `apps/interface/src/views/VideoModelTab.vue` and future video-family branches.

## Notes
- Keep these components presentational only: no stores, no family composables, no bootstrap/listener side effects.
- Family-specific runtime ownership stays under `apps/interface/src/views/video-model/**`.
- Shared styling for these components lives in `apps/interface/src/styles/components/video-generation-cards.css`.
