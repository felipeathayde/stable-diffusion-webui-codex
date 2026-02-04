<!-- tags: frontend, components, ui, primitives -->
# apps/interface/src/components/ui Overview
Date: 2025-12-23
Last Review: 2026-02-04
Status: Active

## Purpose
- Small, reusable UI primitives used across views/components (modals, form widgets).

## Key Files
- `apps/interface/src/components/ui/Modal.vue` — Generic modal shell used by views (file browser, overrides, etc).
- `apps/interface/src/components/ui/SliderField.vue` — Shared slider layout (label left, input right, slider below).
- `apps/interface/src/components/ui/NumberStepperInput.vue` — Numeric input with optional +/- stepper controls.
- `apps/interface/src/components/ui/DimensionPresetsGrid.vue` — Resolution preset buttons (2×2 grid), used by `BasicParametersCard.vue`.
- `apps/interface/src/components/ui/UpscalerTileControls.vue` — Tile presets + overlap + min tile + explicit OOM fallback toggle (shared by hires-fix + `/upscale`).
- `apps/interface/src/components/ui/JsonTreeView.vue` — Collapsible JSON renderer used by the metadata modal (`<details>/<summary>` tree).
- `apps/interface/src/components/ui/Dropzone.vue` — Drag-and-drop file picker primitive (presentational; emits `select`/`rejected`).

## Notes
- Keep components presentational: props in, emits out; no store calls or API fetching.
- Styling should live under `apps/interface/src/styles/components/` and use `cdx-*` semantic classes.
- 2026-01-03: Added standardized file header blocks to UI primitive components (doc-only change; part of rollout).
- 2026-01-13: `JsonTreeView.vue` supports expand/collapse-all signals (used by the metadata modal controls).
- 2026-01-29: Added `Dropzone.vue` with styles in `apps/interface/src/styles/components/cdx-dropzone.css`.
- 2026-02-04: `UpscalerTileControls.vue` now exposes `min_tile` as an Advanced control (keeps backend tile fallback behavior visible and configurable).
