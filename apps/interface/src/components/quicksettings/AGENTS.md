# apps/interface/src/components/quicksettings Overview
<!-- tags: frontend, quicksettings, engines -->
Date: 2025-12-06
Owner: Frontend Maintainers
Last Review: 2026-01-05
Status: Active

## Purpose
- Compact engine/paths/performance controls rendered in the main header (`QuickSettingsBar.vue`), backed by the global `quicksettings` store.

## Key Files
- `QuickSettingsBase.vue` — Generic quicksettings (mode, checkpoint, VAE, optional text encoder) for SD15/SDXL model tabs; advanced controls are rendered by `QuickSettingsBar.vue`.
- `QuickSettingsPerf.vue` — Performance toggles shared across engines (Smart Offload/Fallback/Cache/Core Streaming) rendered in the Advanced nested area.
- `QuickSettingsWan.vue` — WAN22-specific quicksettings (mode + `LightX2V` toggle button, high/low model dirs, metadata/text encoder/VAE selectors).
- `QuickSettingsFlux.vue` / `QuickSettingsZImage.vue` — FLUX.1/ZImage-specific checkpoint/VAE/text encoder selectors (advanced controls are rendered by `QuickSettingsBar.vue`).

## Notes
- `QuickSettingsBase` stays presentational and engine-agnostic; engine-specific filtering and labels (e.g. FLUX.1-only TE layout, WAN-only selectors) live in `QuickSettingsBar.vue`.
- `QuickSettingsBar.vue` renders a main row for engine selectors and a collapsible Advanced nested area (Smart toggles + GPU VRAM / Attention Backend / Overrides), with a left-side handle button.
- `QuickSettingsPerf` uses toggle buttons (`.qs-toggle-btn`) for Smart Offload/Fallback/Cache/Core Streaming (no legacy switches).
- Text encoder dropdowns display a compact label (`family/basename`) even when `/api/text-encoders` or the inventory return long absolute paths; the full value is still posted back in the `<option value>`.
- For FLUX.1, `QuickSettingsBar` hides the base text encoder field and exposes a FLUX.1-only pair of text encoder selectors based on individual files under `flux1_tenc`; wiring to backend overrides is intentionally deferred to a dedicated handoff.
- 2025-12-27: Removed the `hideCheckpoint` toggle/prop; checkpoint selection is always rendered, and on `/models/:tabId` it is tab-scoped (`tab.params.checkpoint`, auto-seeded) while still filtering choices by engine-specific `*_ckpt` roots from `apps/paths.json` (plus user-added paths).
- 2025-12-14: WAN text encoder selector now lists explicit `.safetensors` files under `wan22_tenc` and stores values as `wan22/<abs_path>` for consistent labeling; payload builders must normalize before sending to backend.
- 2025-12-14: WAN Metadata/VAE selectors now prefer concrete inventory paths (VAE constrained by `wan22_vae`), keeping the video endpoints strict about asset paths.
- 2025-12-15: QuickSettings WAN groups now use `.qs-group-wan-*` sizing hooks so the header flex layout doesn’t collapse all controls to the left on wide screens.
- 2025-12-15: WAN “Browse…” actions in `QuickSettingsWan.vue` are rendered as compact `+` icon buttons to match the header quicksettings affordance.
- 2025-12-20: Replaced WAN “Format” with a `LightX2V` toggle; per-stage LoRA selection now lives in the WAN tab (High/Low Noise) when enabled.
- 2025-12-26: Removed the WAN Assets modal; metadata/text encoder/VAE selectors are now inline in the header quicksettings bar.
- 2025-12-26: QuickSettings buttons now use `qs-btn-secondary`/`qs-btn-outline` so they fill the `qs-row` height and keep a visible border (no fixed `2rem` height from `.btn-*` variants).
- 2025-12-28: Removed the obsolete “Diffusion in Low Bits” selectors and moved Smart toggles + GPU VRAM / Attention Backend / Overrides into a collapsible Advanced nested area (open by default, left-side handle); WAN `LightX2V` is a toggle button and the Guided gen header button is hidden for now.
- 2025-12-31: `QuickSettingsWan.vue` now declares `defineEmits(...)` for `browse*` + `update:*` events to avoid Vue “extraneous non-emits listeners” warnings with a fragment root template.
- 2026-01-03: Added standardized file header blocks to quicksettings components (doc-only change; part of rollout).
