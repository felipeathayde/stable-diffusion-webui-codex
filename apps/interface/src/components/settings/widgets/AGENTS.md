<!-- tags: frontend, settings, widgets, paths -->
# apps/interface/src/components/settings/widgets Overview
Date: 2026-01-03
Last Review: 2026-01-03
Status: Active

## Purpose
- Small, reusable Settings widgets used by `apps/interface/src/components/settings/**`.

## Key Files
- `PathList.vue` — Editable list widget for path arrays used by `SettingsPaths.vue`.

## Notes
- Keep widgets presentational: props in, emits out; do not call stores or backend APIs here.
