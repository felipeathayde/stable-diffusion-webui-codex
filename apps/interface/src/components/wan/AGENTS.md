<!-- tags: frontend, components, wan22, video -->
# apps/interface/src/components/wan Overview
Date: 2025-12-14
Last Review: 2026-01-27
Status: Active

## Purpose
- WAN22-specific, presentational components used by the WAN model tab (`WANTab.vue`).

## Key Files
- `WanStagePanel.vue` — High/Low stage controls (sampler/scheduler/steps/cfg/seed + optional per-stage LoRA when `LightX2V` is enabled).
- `WanStageLoraField.vue` — Stage-level LoRA select + weight (wan22-loras), used by `WanStagePanel.vue`.
- `WanSubHeader.vue` — Small section sub-header used by `WANTab.vue` to keep “Video / High / Low” headers consistent.
- `WanVideoOutputPanel.vue` — Video export + interpolation controls (format/crf/pix_fmt/loop/pingpong/save + return-frames + RIFE).

## Notes
- Keep these components dumb: props in, emits out. Do not fetch inventory or call backend APIs here.
- Prefer shared selectors (`SamplerSelector.vue`, `SchedulerSelector.vue`) over ad-hoc `<select>` blocks.
- 2025-12-15: Panels were restyled to render as cards (`.gen-card`) inside WANTab’s “Generation Parameters” panel, using responsive `.wan22-grid` + `.wan22-toggle*` classes.
- 2025-12-20: Removed stage-level “Lightning/Use LoRA” checkboxes; LoRA selection is now a per-stage `<select>` shown only in `LightX2V` mode (WAN QuickSettings).
- 2025-12-22: `WanStagePanel.vue` now uses SDXL-style sliders + steppers for Steps/CFG and moves seed actions (🎲/↺) inside the seed input.
- 2025-12-22: `wan22-settings.css` switches `.wan22-grid` to flex-wrap and adds `wan22-field--{sm,wide,full}` sizing helpers; WAN panels now use these helpers to keep sliders readable without hard grid columns.
- 2025-12-23: WAN panels now use shared gen-card layout primitives (`gc-row`, `gc-col`, `row-split`, `cdx-form-row`) and the new `gen-card--embedded` variant, reducing `wan22-*` layout classes in the tab UI.
- 2025-12-23: `WanStagePanel.vue` renders Steps/CFG via `components/ui/SliderField.vue` (label+input header, slider below) for parity with the rest of the WebUI.
- 2025-12-23: WAN sliders use `cdx-input-w-md` sizing (removes WAN-only `w-step/w-cfg` CSS).
- 2025-12-26: `WanStagePanel.vue` now places Sampler/Scheduler/Steps on the first row and Seed/CFG on the second; LoRA UI was extracted into `WanStageLoraField.vue`.
- 2025-12-28: Added `WanSubHeader.vue` and made `WanVideoOutputPanel.vue` embeddable (so WANTab can compose “Video Output” without nested card borders); Interpolation (RIFE) is now a single toggle button.
- 2025-12-29: `WanVideoOutputPanel.vue` renders the RIFE toggle inline with the other output toggles (Ping-pong/Save/Trim) for layout parity.
- 2026-01-03: Added standardized file header blocks to WAN components (doc-only change; part of rollout).
- 2026-01-06: `WanStagePanel.vue` now labels empty sampler/scheduler selections as “Inherit” (stage overrides are optional; no automatic token).
- 2026-01-27: Added a `Return frames` toggle to `WanVideoOutputPanel.vue` (default off) and an inline note when `Save output` is off (frames still returned so users can download them).
- 2026-02-20: `WanSubHeader.vue` now supports opt-in full-row toggle behavior (`clickable` + `header-click`), with built-in interactive-target exclusion and Enter/Space keyboard parity for collapsible cards.
