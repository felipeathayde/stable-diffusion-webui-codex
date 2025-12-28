# apps/interface/src/views Overview
<!-- tags: frontend, views, txt2img, sdxl -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-27
Status: Active

## Purpose
- Page-level Vue components mapped to routes (e.g., generation workspace, settings).

## Notes
- Views should compose reusable components and stores; avoid duplicating logic that belongs in shared modules.
- Keep routes documented in `router.ts` and the UI taxonomy in `.sangoi/frontend/guidelines/`.
- `Home.vue` is the engine-agnostic landing page; it explains the overall layout (Home, SDXL, Model Tabs, Workflows), links to `.sangoi/**` docs and tasks, and renders Markdown help snippets from `apps/interface/public/help/*.md` via `MarkdownHelp.vue`.
- `Sdxl.vue` reuses the txt2img layout with the SDXL store; keep both views in sync when adjusting shared UX patterns (including prompt modals like LoRA/TI selectors). A gentime badge ao lado do botão Generate mostra o tempo aproximado entre o clique e a chegada do primeiro resultado, alimentado por `store.gentimeMs`.
- Txt2Img/Sdxl now render a nested “Hires Refiner” block inside the Highres card while keeping the global Refiner card under Generation Parameters.
- `XyzPlot.vue` adds an XYZ sweep page (route `/xyz`) that drives batched txt2img runs from frontend only; uses current SDXL state as the baseline payload.
- 2025-12-14: `WANTab.vue` uses typed (Zod) WAN video payload builders and surfaces streaming progress + returned `info` JSON to reduce request drift during WAN22 pipeline testing.
- 2025-12-14: `WANTab.vue` was de-bloated by extracting High/Low + Output panels into `components/wan/*` and treating WAN assets (model dirs/TE/VAE) as QuickSettings responsibility (tab shows a summary + validates before starting runs).
- 2025-12-15: `WANTab.vue` parameters UI was reshaped to mirror Txt2Img’s layout (Prompt + Generation Parameters with cards).
- 2025-12-16: `WANTab.vue` adds a `vid2vid` mode (video upload + flow-chunks params) and shows exported video playback when the backend returns `video.rel_path` from `/api/vid2vid`.
- 2025-12-17: `WANTab.vue` moves Mode/Format selection into QuickSettings, adds “Guided gen” (pulse + tooltip focus for missing prerequisites), hides the Input card in txt2vid, removes “WAN Runtime & Assets” and the bottom Workflows panel, and adds a Low Noise “Use High settings” toggle. `WorkflowsList.vue` now uses a shared store and correctly renders workflow fields (`source_tab_id`, `created_at`).
- 2025-12-20: `WANTab.vue` replaces WAN “Format” with a `LightX2V` quicksetting; when enabled, High/Low Noise show per-stage LoRA selects (from `wan22-loras`) and the UI relies on backend auto-detect for model format (no forced `wan_format`).
- 2025-12-22: `WANTab.vue` now uses sliders for core video params (width/height/frames/fps) and stage params (steps/CFG); seed actions (🎲/↺) are embedded inside the seed input.
- 2025-12-23: `WANTab.vue` layout now relies on shared gen-card primitives (`gc-row`, `gc-col`, `row-split`, `cdx-form-grid`, `cdx-form-row`) instead of WAN-specific grid/toggle classes.
- 2025-12-23: `WANTab.vue` prompt header now includes the same toolbar actions as SDXL (TI/LoRA/Styles), and Low Noise uses a header switch + explicit collapse control (starts expanded).
- 2025-12-25: Prompt panels now use `components/prompt/PromptCard.vue` (shared toolbar + modals). Negative prompt is hidden by default across most views; SDXL shows it by default.
- 2025-12-25: `Txt2Img.vue`, `Sdxl.vue`, `Flux.vue`, `ZImage.vue`, and `WANTab.vue` Results panels now use `components/results/ResultsCard.vue` (3-col header) and reuse `formatJson` from `composables/useResultsCard` for info JSON rendering.
- 2025-12-26: `Txt2Img.vue`, `Sdxl.vue`, `Flux.vue`, and `ZImage.vue` now render “Generation Parameters” via `BasicParametersCard.vue` (WAN-style row grouping).
- 2025-12-26: Panel headers are now consistent (no title wrapper `<span>`); `.panel-header` min height is `2.75rem` (see `apps/interface/src/styles.css`).
- 2025-12-27: `Txt2Img.vue`, `Sdxl.vue`, `Flux.vue`, `ZImage.vue`, `WANTab.vue`, and `ImageModelTab.vue` now render a `Run` card above `Results` (`RunCard.vue`): Generate lives in the Run header; image views use a batch dropdown (applies immediately; OK closes); run summary lines render as chips; SDXL/Flux/ZImage surface “Save Profile” in the “Generation Parameters” header; WAN and image model tabs expose “Save snapshot” + “Copy params” in the Results header (Results header is non-sticky; Run is the single sticky bar).
- 2025-12-23: Sliders across views now use the shared `components/ui/SliderField.vue` layout (label+input header, slider below).
- 2025-12-23: Slider numeric inputs now use shared `cdx-input-w-{xs,sm,md}` widths (removes view/card-specific width hacks).
- 2025-12-14: Removed the legacy standalone `Txt2Vid.vue` view; WAN video entry stays exclusively under model tabs (`/models/:tabId` with `type === 'wan'`).
- 2025-12-14: `ModelTabView.vue` keys per-tab views by `tab.id` so switching `/models/:tabId` remounts the correct tab implementation (prevents composables binding a stale id).
- 2025-12-15: Added `RedirectToModelTab.vue` and router aliases so legacy nav paths can redirect into `/models/:tabId` (WAN) without spamming Vue Router “No match found” warnings.
- 2025-12-15: Model-tab actions (rename/enable/load/unload/duplicate/remove) were moved out of the per-tab view; `Home.vue` is now the canonical place to manage tabs, and the per-tab “Send to Workflows” action lives in a dedicated panel under Generation Parameters.
- 2025-12-19: `ToolsTab.vue` GGUF Converter expanded with SoTA presets (`Q4_K_M/Q5_K_M`), additional quant types (`Q2_K/Q3_K/IQ4_NL/Q6_K` + legacy `Q4_0/Q4_1/Q5_0/Q5_1`), and an advanced per-tensor override textarea; default quantization is now `Q5_K_M`.
- 2025-12-22: `ToolsTab.vue` UI was migrated off a large `<style scoped>` block into `apps/interface/src/styles/views/tools.css`; the file browser now uses `components/ui/Modal.vue` and progress uses a native `<progress>` element (no inline width styles).
- 2025-12-22: `Home.vue`, `ModelsList.vue`, and `WorkflowsList.vue` were cleaned up to use shared primitives + `cdx-list` (no legacy `muted/list/grid-*` classes, no `style="..."` attributes). `XyzPlot.vue` no longer uses inline styles for its results grid.
- 2025-12-27: “Send to Img2Img/Inpaint” actions from Txt2Img/SDXL/Flux now open the matching model tab (`/models/:tabId`) and set `useInitImage` + `initImageData` (removes broken navigation to a nonexistent `/img2img` SPA route). `EngineView.vue` now fails fast with an explicit “Engine not implemented” panel when no view is registered and sets `codex_engine` via QuickSettings when the engine is implemented.
