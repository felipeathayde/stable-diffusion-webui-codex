<!-- tags: frontend, components, prompt, highres, refiner -->
# apps/interface/src/components Overview
Date: 2025-12-06
Owner: Frontend Maintainers
Last Review: 2026-01-13
Status: Active

## Purpose
- Reusable Vue components (panels, form widgets, prompts, galleries) shared across views.

## Notes
- Components should be presentational and rely on Pinia stores or props for state.
- Follow the styling rules documented in `.sangoi/frontend/guidelines/frontend-style-guide.md`.
- Prompt parsing/serialization lives in `prompt/PromptToken.ts` with Vitest coverage; ensure new prompt widgets pass through that module.
- Generation + highres + refiner controls live in `GenerationSettingsCard.vue`, `HighresSettingsCard.vue`, and `RefinerSettingsCard.vue`, all using CSS grid layouts.
- Model/sampler/scheduler dropdowns vivem em `ModelSelector.vue`, `SamplerSelector.vue` e `SchedulerSelector.vue`; views devem reutilizar esses componentes em vez de construir selects ad-hoc. Presets/estilos são tratados hoje pelas próprias views (SDXL/FLUX.1) sem um componente dedicado de selector.
- `QuickSettingsBar.vue` surfaces engine/tab selectors in the main header row; it renders a nested, collapsible Advanced area (Smart toggles + GPU VRAM / Attention Backend / Overrides) with a left-side handle. In `/models/:tabId`, the active family comes from the tab type; outside model tabs, it falls back to `quicksettings.currentEngine`.
- 2026-01-13: Metadata modal payload now uses a single `metadata` object (no `codex_metadata` wrapper; `file_metadata` → `metadata`).
- 2026-01-13: Metadata modal adds a toggle to switch between raw (flat) and nested (organized) views for file metadata.
- 2026-01-13: Checkpoint metadata payload uses `file.{name,path,size}` (no `title`/`model_name`/`filename`) and the file-metadata keys are normalized (e.g. `model.*`, `codex.*`, `gguf.*`).
- 2025-12-29: `QuickSettingsBar.vue` no longer writes `--sticky-offset` directly; the header offset is tracked by `App.vue` via a `ResizeObserver`.
- 2025-12-29: `QuickSettingsBar.vue` keeps the active model tab in sync with the current route (`/models/:tabId`) to avoid falling back to the global engine during Vite HMR reloads.
- 2025-12-27: `QuickSettingsBar.vue` binds checkpoint selection to the active model tab (`tab.params.checkpoint`, auto-seeded from the engine’s `*_ckpt` roots in `apps/paths.json`), and FLUX.1/ZImage model tabs also keep per-tab text encoders (`tab.params.textEncoders`) used by `useGeneration` for `tenc_sha`/`text_encoder_override`.
- 2026-01-01: `QuickSettingsBar.vue` “Refresh” now triggers a checkpoint rescan (`/api/models?refresh=1`) so newly copied weights under `*_ckpt` roots show up without restarting the backend.
- 2025-12-26: QuickSettings header buttons now use `qs-btn-secondary`/`qs-btn-outline` (fill the `qs-row` height, with consistent borders; no fixed `2rem` height).
- `ResultViewer.vue` exibe um overlay full-screen para zoom de imagens (sem modal encaixotado): o preview da galeria continua grande no card, enquanto o overlay usa o viewport inteiro com ferramenta lateral para pan/zoom (drag para pan, botões de Fit/1:1/+/−/Close na barra à direita).
- 2025-12-29: `ResultViewer.vue` now renders the zoom overlay inside the `.viewer-card` root so fallthrough attrs like `:style`/`class` can be applied without Vue fragment-root warnings.
- 2026-01-01: `ResultViewer.vue` can optionally show a single `previewImage` (with `previewCaption`) while a task is running, before final results are available.
- 2025-12-16: Added `InitialVideoCard.vue` to mirror `InitialImageCard.vue` for WAN `vid2vid` uploads (file picker + preview + remove).
- 2025-12-17: `QuickSettingsWan.vue` added WAN Mode/Format selectors + Guided gen; `QuickSettingsBar.vue` dispatches WAN events (`codex-wan-mode-change`, `codex-wan-guided-gen`) consumed by `WANTab.vue`.
- 2025-12-28: QuickSettings now groups GPU VRAM / Attention Backend / Overrides into a collapsible Advanced row; the obsolete low-bits dtype selectors were removed and Guided gen entrypoint is hidden for now.
- 2025-12-14: QuickSettings WAN text encoder dropdown now prefers concrete `.safetensors` files under `wan22_tenc`, emitting `wan22/<abs_path>` values that the WAN payload builder normalizes before POSTing.
- 2025-12-14: WAN tab UI panels live under `components/wan/` (`WanStagePanel.vue`, `WanVideoOutputPanel.vue`) to avoid duplicating High/Low/Output markup in the view.
- 2025-12-15: `VideoSettingsCard.vue` gained a dedicated stylesheet (`styles/components/video-settings-card.css`) and the WAN tab’s parameter sections were restyled to use card layouts consistently.
- 2025-12-22: `GenerationSettingsCard.vue` now exposes a CFG slider (next to Seed via a flex footer) and moves seed actions (🎲/↺) inside the seed input; `VideoSettingsCard.vue` adds an FPS slider.
- 2025-12-22: `GenerationSettingsCard.vue` internal layout now uses flex rows (`gc-stack`/`gc-row`) so sliders + buttons aren’t constrained by a single grid template.
- 2025-12-22: Removed remaining Vue SFC `<style>` blocks from `SettingsForm.vue` and `ParamBlocksRenderer.vue`; both now rely on `apps/interface/src/styles/components/*` (including `param-blocks.css`) and avoid inline `:style` layout for grids.
- 2025-12-23: Added shared slider primitives (`components/ui/SliderField.vue`, `components/ui/NumberStepperInput.vue`) and migrated sliders to the unified layout (label left + input right above slider).
- 2025-12-23: Deprecated per-card width classes (`w-*`) were removed; use `cdx-input-w-{xs,sm,md}` for numeric sizing.
- 2025-12-25: Added `components/results/ResultsCard.vue` to standardize the 3-column sticky Results header (title / Generate / actions) across generation views.
- 2025-12-26: Added `BasicParametersCard.vue` as a shared “common params” card (sampler/scheduler/steps + seed/CFG + width/height), mirroring WAN stage styling.
- 2025-12-27: `BasicParametersCard.vue` can optionally render resolution presets aligned with the Width/Height controls (`resolutionPresets`, rendered as a 2×2 grid).
- 2025-12-28: QuickSettings perf toggles and other small toggles were unified as `qs-toggle-btn` buttons (replacing the old `.qs-switch` widgets); the Width/Height swap glyph was rotated for legacy parity via `.btn-swap-icon`.
- 2025-12-26: Added `BatchSettingsCard.vue` to keep batch count/size controls as a separate card, and refactored `GenerationSettingsCard.vue` to compose `BasicParametersCard` + `BatchSettingsCard` (backwards-compatible wrapper).
- 2025-12-31: `BasicParametersCard.vue` now supports syncing Width/Height from the init image (`showInitImageDims` + `sync-init-image-dims` event), snaps dimension updates to the input step (default 8; matches backend multiple-of-8 constraint), and raises default max dims to 8192 to accommodate tall portrait inputs.
- 2026-01-01: `BasicParametersCard.vue` can optionally render a `CLIP Skip` control (`showClipSkip`, `clipSkip`, `minClipSkip/maxClipSkip`) so model tabs can expose clip-skip without prompt tags.
- 2026-01-01: `BasicParametersCard.vue` can optionally render a `WanSubHeader` title (`sectionTitle`) so model tabs can label the card like WAN sections.
- 2026-01-02: Added standardized file header docstrings to component modules (doc-only change; part of rollout).
- 2026-01-03: Continued the header rollout across remaining core component modules (doc-only change; part of rollout).
- 2026-01-06: `BasicParametersCard.vue` now defaults to explicit sampler/scheduler selection (no empty option) and selector components tolerate missing `label` by falling back to `name`.
- 2026-01-06: Sampler/Scheduler selectors now default the empty-option label to “Select” (no `Automatic` placeholder); WAN stage panels still override with “Inherit”.
