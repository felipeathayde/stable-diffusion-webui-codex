<!-- tags: frontend, settings, paths -->
# apps/interface/src/components/settings Overview
Date: 2025-12-04
Owner: Frontend Maintainers
Last Review: 2025-12-23
Status: Active

## Purpose
- Settings panels for the Codex WebUI (paths, advanced options) that are shared across engine tabs.

## Notes
- `SettingsPaths.vue` surfaces engine-specific search roots for models/VAEs/LoRAs/Text Encoders by wiring directly to `apps/paths.json` keys (`sd15_*`, `sdxl_*`, `flux_*`, `wan22_*`) via `/api/paths`.
- Keep the UI layout compatible with other settings panels (use `panel-section`, `label-muted`, and shared widgets such as `PathList.vue`).
- When extending settings here, keep DTOs in sync with `apps/interface/src/api/types.ts` and backend routes under `apps/backend/interfaces/api/run_api.py`.
- 2025-12-22: `SettingsForm.vue` no longer uses a Vue SFC `<style scoped>` block; it relies on `apps/interface/src/styles/components/settings-form.css` and uses shared primitives (`select-md`, `slider`, `caption`).
- 2025-12-23: Slider settings now use the shared `components/ui/SliderField.vue` layout (label+input header, slider below).
- 2025-12-23: Settings sliders use `cdx-input-w-sm` for the numeric input width (no more `w-24` one-off CSS).
