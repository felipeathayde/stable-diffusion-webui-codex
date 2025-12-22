# apps/interface/src/components/quicksettings Overview
<!-- tags: frontend, quicksettings, engines -->
Date: 2025-12-06
Owner: Frontend Maintainers
Last Review: 2025-12-22
Status: Active

## Purpose
- Compact engine/paths/performance controls rendered in the main header (`QuickSettingsBar.vue`), backed by the global `quicksettings` store.

## Key Files
- `QuickSettingsBase.vue` — Generic quicksettings row for SD15/SDXL/Flux tabs (mode, checkpoint, VAE, optional text encoder, attention backend, overrides); groups are tagged with `.qs-group-*` classes so the header grid can lay them out in two rows.
- `QuickSettingsPerf.vue` — Performance/runtime controls shared across engines (Diffusion in Low Bits, GPU VRAM limit, Smart Offload/Fallback/Cache switches), rendered on the second row via `.qs-group-perf-*` classes.
- `QuickSettingsWan.vue` — WAN22-specific quicksettings (mode + `LightX2V` toggle, high/low model dirs, WAN assets, guided gen entrypoint, precision, VRAM, attention backend, overrides).

## Notes
- `QuickSettingsBase` stays presentational and engine-agnostic; engine-specific filtering and labels (e.g. Flux-only precision, WAN-only text encoders) live in `QuickSettingsBar.vue`. The bar uses the `.qs-group-*` hooks to place Mode/Checkpoint/VAE/Text Encoder on the first row and Attention/Overrides on the second.
- `QuickSettingsPerf` is intended to sit visually on the second line of the bar, directly under the model/text encoder selectors, and uses switches (`.qs-switch`) instead of plain checkboxes for Smart Offload/Fallback/Cache.
- Text encoder dropdowns display a compact label (`family/basename`) even when `/api/text-encoders` or the inventory return long absolute paths; the full value is still posted back in the `<option value>`.
- For Flux, `QuickSettingsBar` hides the base text encoder field and exposes a Flux-only pair of text encoder selectors based on individual files under `flux_tenc`; wiring to backend overrides is intentionally deferred to a dedicated handoff.
- 2025-12-14: WAN text encoder selector now lists explicit `.safetensors` files under `wan22_tenc` and stores values as `wan22/<abs_path>` for consistent labeling; payload builders must normalize before sending to backend.
- 2025-12-14: WAN Metadata/VAE selectors now prefer concrete inventory paths (VAE constrained by `wan22_vae`), keeping the video endpoints strict about asset paths.
- 2025-12-15: QuickSettings WAN groups now use `.qs-group-wan-*` sizing hooks so the header flex layout doesn’t collapse all controls to the left on wide screens.
- 2025-12-15: WAN “Browse…” actions in `QuickSettingsWan.vue` are rendered as compact `+` icon buttons to match the header quicksettings affordance.
- 2025-12-17: WAN quicksettings adds Mode/Format selectors and a “Guided gen” button; `QuickSettingsBar.vue` dispatches events consumed by `WANTab.vue` to keep the tab state in sync.
- 2025-12-20: Replaced WAN “Format” with a `LightX2V` toggle; per-stage LoRA selection now lives in the WAN tab (High/Low Noise) when enabled.
- 2025-12-22: WAN `LightX2V` control is now a select (`normal`/`LightX2V`) instead of a switch to better match the rest of the header UI.
