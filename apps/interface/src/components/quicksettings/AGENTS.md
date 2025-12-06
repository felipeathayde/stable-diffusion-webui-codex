# apps/interface/src/components/quicksettings Overview
<!-- tags: frontend, quicksettings, engines -->
Date: 2025-12-06
Owner: Frontend Maintainers
Last Review: 2025-12-06
Status: Active

## Purpose
- Compact engine/paths/performance controls rendered in the main header (`QuickSettingsBar.vue`), backed by the global `quicksettings` store.

## Key Files
- `QuickSettingsBase.vue` — Generic quicksettings row for SD15/SDXL/Flux tabs (mode, checkpoint, VAE, text encoder, attention backend, overrides).
- `QuickSettingsPerf.vue` — Performance/runtime controls shared across engines (Diffusion in Low Bits, GPU VRAM limit, Smart Offload/Fallback/Cache switches).
- `QuickSettingsWan.vue` — WAN22-specific quicksettings (high/low model dirs, WAN text encoder/VAE, precision, VRAM, attention backend, overrides).

## Notes
- `QuickSettingsBase` should stay presentational and engine-agnostic; engine-specific filtering and labels (e.g. Flux-only precision, WAN-only text encoders) live in `QuickSettingsBar.vue`.
- `QuickSettingsPerf` is intended to sit visually on the second line of the bar, directly under the model/text encoder selectors, and uses switches (`.qs-switch`) instead of plain checkboxes for Smart Offload/Fallback/Cache.
- Text encoder dropdowns display a compact label (`family/basename`) even when `/api/text-encoders` or the inventory return long absolute paths; the full value is still posted back in the `<option value>`.
- For Flux, `QuickSettingsBar` hides the base text encoder field and exposes a Flux-only pair of text encoder selectors based on individual files under `flux_tenc`; wiring to backend overrides is intentionally deferred to a dedicated handoff.
