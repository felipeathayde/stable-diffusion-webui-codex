# apps/interface/src/components/quicksettings Overview
<!-- tags: frontend, quicksettings, engines -->
Date: 2025-12-06
Owner: Frontend Maintainers
Last Review: 2025-12-06
Status: Active

## Purpose
- Compact engine/paths/performance controls rendered in the main header (`QuickSettingsBar.vue`), backed by the global `quicksettings` store.

## Key Files
- `QuickSettingsBase.vue` — Generic quicksettings row for SD15/SDXL/Flux tabs (mode, checkpoint, VAE, text encoder, precision, VRAM, smart toggles, attention backend, overrides).
- `QuickSettingsWan.vue` — WAN22-specific quicksettings (high/low model dirs, WAN text encoder/VAE, precision, VRAM, attention backend, overrides).

## Notes
- `QuickSettingsBase` should stay presentational and engine-agnostic; engine-specific filtering and labels (e.g. Flux-only precision, WAN-only text encoders) live in `QuickSettingsBar.vue`.
- Text encoder dropdowns display a compact label (`family/basename`) even when `/api/text-encoders` returns long absolute paths; the full label is still posted back in the `value`.
- For Flux, `QuickSettingsBar` hides the base text encoder field and exposes a Flux-only pair of text encoder selectors; only the first entry is currently consumed by `textEncoderOverride`, the second is reserved for future per-component overrides.
