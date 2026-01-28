# apps/interface/src/views Overview
<!-- tags: frontend, views, model-tabs -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2026-01-28
Status: Active

## Purpose
- Page-level Vue components mapped to routes (e.g., generation workspace, settings).

## Notes
- Views should compose reusable components and stores; avoid duplicating logic that belongs in shared modules.
- Keep routes documented in `apps/interface/src/router.ts` and the UI taxonomy in `.sangoi/frontend/guidelines/`.
- All generation workspaces live under model tabs (`/models/:tabId`):
  - `ModelTabView.vue` mounts `WANTab.vue` when `tab.type === 'wan'`.
  - `ModelTabView.vue` mounts `ImageModelTab.vue` when `tab.type` is `sd15|sdxl|flux1|zimage`.
- `Home.vue` is the engine-agnostic landing page and the canonical place to manage tabs (enable/disable, rename, duplicate, remove).
- `WANTab.vue` uses typed WAN video payload builders and `useVideoGeneration(tabId)` for streaming progress.
- `ImageModelTab.vue` mirrors the legacy engine-tab layout (same `panels` + `panel-stack` structure as WAN): PromptCard (progress/error + optional init-image controls), BasicParametersCard + optional Highres/Refiner, RunCard (batch dropdown), ResultsCard (gentime/actions), ResultViewer gallery, and an Info panel; generation/payload wiring lives in `useGeneration(tabId)` and capability gating uses `useEngineCapabilitiesStore()`.
- 2025-12-28: Replaced the remaining small switch widgets (`qs-switch--sm`) in `Home.vue` and `WANTab.vue` with `qs-toggle-btn` buttons for consistent toggle styling across the UI.
- 2025-12-29: `WANTab.vue` uses `WanSubHeader` for consistent section headers and keeps “Video” + “Video Output” as sequential (separate) cards; Video exposes a compact Aspect selector inline with the Width slider.
- 2025-12-29: `WANTab.vue` renders the History card above the results viewer for parity with `ImageModelTab.vue`.
- 2025-12-31: `ImageModelTab.vue` now syncs Width/Height from the init image (auto on upload and “Send to Img2Img”, plus a manual re-sync action) and applies Kontext-friendly defaults on FLUX.1 init-image runs without overriding custom values.
- 2026-01-01: `ImageModelTab.vue` now exposes per-tab `CLIP Skip` (SD15/SDXL/FLUX.1) and persists it in saved profiles; requests send `clip_skip`/`img2img_clip_skip` to the backend.
- 2026-01-01: `ImageModelTab.vue` and `WANTab.vue` History sections now use `WanSubHeader` and hide per-item actions until hover (with horizontal scroll when needed) to save space.
- 2026-01-01: History cards now render as a single-row horizontal strip, and `WANTab.vue` reuses the shared `cdx-history-*` card layout for parity with image tabs.
- 2026-01-01: Results empty states now reflect running tasks (“Starting inference…” / “Generating…”) and image tabs can show live preview frames inside `ResultViewer.vue` while sampling.
- 2026-01-03: Added standardized file header blocks to view modules (doc-only change; part of rollout).
- 2026-01-06: `ImageModelTab.vue` now filters schedulers by the selected sampler’s `allowed_schedulers` and auto-resets invalid scheduler selections to the sampler’s `default_scheduler`.
- 2026-01-13: `ToolsTab.vue` GGUF converter supports cancellation and an Overwrite toggle (default off; fails if the output file exists).
- 2026-01-14: `ToolsTab.vue` GGUF converter supports Comfy/Codex key mapping for diffusion denoisers (Diffusers→Comfy keys via `comfy_layout`).
- 2026-01-14: `ToolsTab.vue` now right-aligns the GGUF converter action rows (toggles + Convert/Cancel).
- 2026-01-16: `ToolsTab.vue` now uses vendored model metadata only and exposes a Mixed toggle with an FP16/FP32 choice (advanced overrides are hidden).
- 2026-01-16: `ToolsTab.vue` GGUF converter presets include WAN22 denoisers (selectable via vendored model metadata).
- 2026-01-17: `WANTab.vue` no longer listens to window `codex-wan-mode-change`; WAN mode presets are applied by `QuickSettingsWan.vue` directly via tab param updates.
- 2026-01-21: WAN stage LoRA selection is sha-based (`loraSha` in params → payload `lora_sha`).
- 2026-01-23: `WANTab.vue` snaps WAN video width/height to multiples of 16 (rounded up; Diffusers parity) so invalid sizes never reach the backend.
- 2026-01-25: `ImageModelTab.vue` CLIP Skip now allows `0` as “use default” (sends `clip_skip=0` so clip-skip state does not leak across jobs).
- 2026-01-27: `WANTab.vue` supports video-first WAN results (exported video shown even when frames are omitted); when `Return frames` is disabled and a video exists, the frames viewer empty state shows “Frames not returned” with a hint.
- 2026-01-28: `ImageModelTab.vue` treats Z-Image as Turbo/Base variant-dependent: toolbar/CFG labels and negative-prompt support follow the per-tab Turbo toggle.
