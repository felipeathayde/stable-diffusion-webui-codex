<!-- tags: frontend, components, wan22, video -->
# apps/interface/src/components/wan Overview
Date: 2025-12-14
Owner: Frontend Maintainers
Last Review: 2025-12-15
Status: Active

## Purpose
- WAN22-specific, presentational components used by the WAN model tab (`WANTab.vue`).

## Key Files
- `WanStagePanel.vue` — High/Low stage controls (sampler/scheduler/steps/cfg/seed + Lightning/LoRA).
- `WanVideoOutputPanel.vue` — Video export + interpolation controls (format/crf/pix_fmt/loop/pingpong/save flags + RIFE).

## Notes
- Keep these components dumb: props in, emits out. Do not fetch inventory or call backend APIs here.
- Prefer shared selectors (`SamplerSelector.vue`, `SchedulerSelector.vue`) over ad-hoc `<select>` blocks.
- 2025-12-15: Panels were restyled to render as cards (`.gen-card`) inside WANTab’s “Generation Parameters” panel, using responsive `.wan22-grid` + `.wan22-toggle*` classes.
