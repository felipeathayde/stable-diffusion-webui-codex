<!-- tags: frontend, components, wan22, video -->
# apps/interface/src/components/wan Overview
Date: 2025-12-14
Owner: Frontend Maintainers
Last Review: 2025-12-23
Status: Active

## Purpose
- WAN22-specific, presentational components used by the WAN model tab (`WANTab.vue`).

## Key Files
- `WanStagePanel.vue` — High/Low stage controls (sampler/scheduler/steps/cfg/seed + optional per-stage LoRA when `LightX2V` is enabled).
- `WanVideoOutputPanel.vue` — Video export + interpolation controls (format/crf/pix_fmt/loop/pingpong/save flags + RIFE).

## Notes
- Keep these components dumb: props in, emits out. Do not fetch inventory or call backend APIs here.
- Prefer shared selectors (`SamplerSelector.vue`, `SchedulerSelector.vue`) over ad-hoc `<select>` blocks.
- 2025-12-15: Panels were restyled to render as cards (`.gen-card`) inside WANTab’s “Generation Parameters” panel, using responsive `.wan22-grid` + `.wan22-toggle*` classes.
- 2025-12-20: Removed stage-level “Lightning/Use LoRA” checkboxes; LoRA selection is now a per-stage `<select>` shown only in `LightX2V` mode (WAN QuickSettings).
- 2025-12-22: `WanStagePanel.vue` now uses SDXL-style sliders + steppers for Steps/CFG and moves seed actions (🎲/↺) inside the seed input.
- 2025-12-22: `wan22-settings.css` switches `.wan22-grid` to flex-wrap and adds `wan22-field--{sm,wide,full}` sizing helpers; WAN panels now use these helpers to keep sliders readable without hard grid columns.
- 2025-12-23: WAN panels now use shared gen-card layout primitives (`gc-row`, `gc-col`, `row-split`, `cdx-form-row`) and the new `gen-card--embedded` variant, reducing `wan22-*` layout classes in the tab UI.
- 2025-12-23: `WanStagePanel.vue` aligns slider rows with shared field structure (`field` + `ml-steps`) for consistent spacing with SDXL.
