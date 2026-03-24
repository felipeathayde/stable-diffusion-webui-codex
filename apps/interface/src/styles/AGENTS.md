<!-- tags: frontend, styles, tokens, conventions -->
# apps/interface/src/styles Overview
Date: 2025-12-22
Last Review: 2026-03-16
Status: Active

## Purpose
- Define the Codex WebUI design language (tokens + primitives) and keep CSS consistent across views.
- Keep templates readable by avoiding “utility soup” and one-off styling.
- Make styling changes predictable: tokens first, shared primitives second, feature CSS last.

## Stack (how CSS ships)
- Entrypoint: `apps/interface/src/styles.css` (imported by `apps/interface/src/main.ts`).
- Tailwind v4 is enabled via `@tailwindcss/vite` (see `apps/interface/vite.config.ts`) and compiled through `@import "tailwindcss"` in `styles.css`.
- Every stylesheet under `apps/interface/src/styles/**` is pulled in via `styles.css` (imports live inside `@layer components`).
- No shadow DOM / `?inline` injection in this repo.

## Where files go
- `apps/interface/src/styles.css`
  - Canonical tokens (`:root`, `.dark`) + global primitives (`.btn`, `.panel`, `.ui-input`, `.select-md`, `.slider`, etc.).
  - Imports for `src/styles/components/*.css` and `src/styles/views/*.css` (inside `@layer components`).
- `apps/interface/src/styles/components/*.css`
  - Reusable component-level styles (scoped by semantic classes).
- `apps/interface/src/styles/views/*.css`
  - View-specific styling/layout (WAN, Settings, PNG Info, etc.).
- `apps/interface/src/styles/EXAMPLE-dashboard-surface-*.css`
  - Reference-only examples for CSS structure (not imported into the build).

## Tokens & Variables (Codex)
Canonical tokens live in `apps/interface/src/styles.css` under `:root` and `.dark`:
- Spacing/layout: `--space-{xs,sm,md,lg,xl}`, `--container-px`, `--content-px`
- Control geometry: `--control-h-{sm,md,lg}`
- Theme palette: `--background`, `--foreground`, `--panel`, `--accent`, `--border`, etc (light + `.dark`)
- Tailwind mapping: `@theme` maps these into `--color-*` variables used by primitives.

Rules:
- Don’t hard-code colors/shadows in feature CSS. Use existing `--color-*` variables or add a token.
- When adding a new token, define it in `styles.css` and keep dark-mode parity.
- For new component/view-scoped tokens, use a `cdx` prefix: `--cdx-<scope>-<name>`.

## Naming (replace “ab” → “cdx”)
- Avoid creating new unprefixed “global-ish” classes. This repo already has legacy primitives (`.btn`, `.panel`, `.ui-input`, …) — don’t create more.
- New shared / reusable semantic classes should use the `cdx-` prefix.
  - BEM is allowed: `cdx-card__header`, `cdx-card--dense`.
- Existing feature prefixes are allowed where established (`qs-*`, `hr-*`, `rf-*`, `wan-*`, `wan22-*`), but avoid inventing new ad-hoc prefixes.
- State should be semantic: `.is-open`, `.is-active`, `.has-error` or `data-*`.

## Template rules (avoid “inline styling”)
- No `style="..."` in templates and no direct `el.style.*` mutations.
- Prefer semantic classes that map to rules in `src/styles/**`.
- Tailwind utilities in templates are fine for small glue, but avoid long utility chains (“utility soup”).
  - If a layout pattern repeats, promote it to a semantic class and style it in the right stylesheet (you can also use `@apply` there).

## Vue SFC `<style>` blocks
- Avoid adding new `<style>` / `<style scoped>` blocks for UI styling; prefer `apps/interface/src/styles/**`.
- If you touch a legacy `<style>` block, consider migrating it into `src/styles/components/<component>.css` (and importing via `styles.css`).

## Adding a new stylesheet
1. Search before adding: `rg -n "<selector-or-token>" apps/interface/src/styles`
2. Create a file:
   - Component: `apps/interface/src/styles/components/<component>.css`
   - View: `apps/interface/src/styles/views/<view>.css`
3. Import it from `apps/interface/src/styles.css` (inside `@layer components`) and keep ordering stable.

## Validate
- `cd apps/interface && npm run build`
- `cd apps/interface && npm run typecheck`

## Reference-only examples (not shipped)
- `apps/interface/src/styles/EXAMPLE-dashboard-surface-base.css`
- `apps/interface/src/styles/EXAMPLE-dashboard-surface-theme.css`

These reference files may contain `ab-*` / `--ab-*` from the source project; when porting ideas, adapt naming to Codex (`cdx-*`, `--cdx-*`) and keep implementation consistent with this repo’s pipeline.

## Recent updates
- 2025-12-22: `styles/components/generation-settings-card.css` now uses flex rows (`gc-stack`/`gc-row`) so card internals aren’t bound to a single grid template.
- 2025-12-23: Added shared layout helpers `cdx-form-grid`/`cdx-form-row` (in `styles/components/views-shared.css`) and a `gen-card--embedded` variant for cardless embedded layouts.
- 2025-12-23: Added shared slider primitives styling (`styles/components/cdx-slider-field.css`, `styles/components/cdx-stepper-input.css`).
- 2025-12-23: Removed legacy `w-*` width classes and `!important` sizing rules; numeric inputs should use `cdx-input-w-{xs,sm,md}`.
- 2026-01-13: `styles/components/asset-metadata-modal.css` adds in-view controls + subtitle styling for the metadata modal (Beautify + expand/collapse all).
- 2025-12-27: Added shared “Run header” layout helpers (`run-controls`, `run-control`, `run-sticky`) in `styles/components/views-shared.css` (Run is the single sticky header in generation views).
- 2025-12-27: Added `run-batch-menu*` styles in `styles/components/views-shared.css` so Run batch count/size controls can live in a dropdown panel.
- 2026-03-02: `styles/components/views-shared.css` now includes a dedicated `run-badge-xyz` style (cyan/blue identity, distinct from experimental orange badges) used beside `Generate` in image-tab RUN headers when XYZ workflow is enabled.
- 2025-12-27: Added shared helpers for input-inline actions and chip rows (`cdx-input-with-actions`, `cdx-chips-row`) and a `cdx-res-presets` block for aligning size presets with Width/Height (2×2 grid).
- 2025-12-27: Prompt toolbar Styles field now has a min width to prevent the inline actions from overlapping the input text.
- 2025-12-26: Standardized `.panel-header` height to `4.25rem` (min height) and removed title-wrapper `<span>` patterns from panel headers.
- 2025-12-26: QuickSettings buttons now use `qs-btn-secondary`/`qs-btn-outline` so they fill the `qs-row` height with consistent borders (no fixed `2rem` height).
- 2025-12-28: QuickSettings toggles now use `qs-toggle-btn` (neon border states) and the legacy `qs-switch` styling was removed.
- 2026-02-08: `quicksettings.css` now styles a paired mode-toggle group (`IMG2IMG`/`INPAINT`) in the top row and applies explicit disabled-state treatment for `qs-toggle-btn:disabled`.
- 2026-02-15: Added apply-status note styles for settings saves (`settings-form.css`) and quicksettings override apply-state messaging (`quicksettings-overrides-modal.css`), replacing static restart-only wording.
- 2025-12-28: QuickSettings bar now stacks `quicksettings-row` blocks and uses an animated (rAF) collapsible Advanced row; WAN adds `wan-subheader` section headers in `styles/views/wan.css`.
- 2025-12-29: Sticky header offset (`--sticky-offset`) is derived from the `.main-header` height and used by `RunCard` (`.panel-header.results-sticky`).
- 2026-01-01: History cards now render as a single-row horizontal strip (with horizontal scroll); per-item action buttons appear on hover as a compact overlay (with horizontal scroll when needed) and stay visible for the selected item via `styles/components/views-shared.css` (image tabs + WAN).
- 2026-03-02: Tools view keeps GGUF action rows right-aligned for `Overwrite` and `Convert/Cancel` controls via `styles/views/tools.css`.
- 2026-01-29: Added `styles/components/cdx-dropzone.css` and `styles/views/pnginfo.css` for the revamped PNG Info view/dropzone.
- 2026-02-18: `styles/views/pnginfo.css` now defines a PNG-specific header-actions layout (`pnginfo-header-*`) to avoid shared `results-actions` row-wrap collisions, and adds loaded-preview clear-button positioning (`pnginfo-clear-button`).
- 2026-02-03: Renamed the hires settings card stylesheet to `styles/components/hires-settings-card.css` and updated the card root class to `.hires-card`.
- 2026-02-08: Added `styles/components/img2img-inpaint-params-card.css` for extracted img2img/inpaint controls and aligned `hires-settings-card.css` / `refiner-settings-card.css` layout structure to the `gen-card` organization style.
- 2026-02-08: `hires-settings-card.css` now styles the full Basic-params-like hires row layout (including pseudo-disabled scale visuals when width/height overrides are active); tile preset visuals in hires use the shared resolution-button variant from `UpscalerTileControls.vue`.
- 2026-02-18: Added `styles/components/img2img-basic-parameters-card.css` for the new init-image basic-parameters card and imported it in `styles.css`.
- 2026-02-18: Hires card row order parity update (dimensions row before upscaler row) keeps selectors scoped under `.hires-card` / `.img2img-basic-card` to avoid global `.gc-row` regressions.
- 2026-02-18: `generation-settings-card.css` now includes scoped CFG-advanced row styles (`.cfg-advanced-row`) and CFG header toggle spacing to support the new Advanced guidance controls without affecting non-generation panels.
- 2026-02-18: Added `styles/components/cdx-hover-tooltip.css` and imported it in `styles.css`; slider labels can now render a polished hover/focus tooltip panel via `HoverTooltip.vue`.
- 2026-02-17: Added `styles/components/xyz-sweep-card.css` and removed `styles/views/xyz.css`; XYZ styling now lives with the shared embeddable card.
- 2026-02-17: Added shared `run-progress-status*` rules in `styles/components/views-shared.css` and removed duplicated WAN-only progress styles from `styles/views/wan.css`.
- 2026-02-17: `quicksettings.css` flex sizing was adjusted for wide monitors (reduced right-side dead space in the top row).
- 2026-02-21: `quicksettings.css` removed the obsolete `.qs-group-attention` layout hook after attention backend control moved from QuickSettings to Launcher Runtime.
- 2026-02-22: `quicksettings.css` adds `.qs-group-perf-obliterate` to the perf-group flex layout so the new `Obliterate VRAM` quicksettings action aligns with other Smart toggles.
- 2026-03-02: `quicksettings.css` now supports right-anchored family controls (`qs-group-mode-toggle--end`, `qs-group-wan-refresh--end`) so ZImage mode toggles sit before Refresh on the right edge and WAN Refresh aligns to the right edge.
- 2026-02-17: `result-viewer.css` removed zoom legend styles; zoom UI now relies on icon-only controls in the shared overlay.
- 2026-03-01: `result-viewer.css` now styles WAN zoom frame-guide editing UX in `ImageZoomOverlay.vue` (conditional wider toolbar for guide mode, draggable frame-guide rectangle state, guide resize-mode/size controls, and compact source/scaled/frame/crop metadata rows).
- 2026-03-02: `styles.css` now positions `InitialImageCard` dropzone remove action inside the dotted zone (`.init-dropzone-remove` absolute top-right), and `img2img-inpaint-params-card.css` adds `.img2img-caption--init-name` for centered init-image filename captions. On 2026-03-24 that same stylesheet also gained the compact inpaint preview legend chips that label the mask, blur range, and final blue crop box under the thumbnail.
- 2026-03-02: `styles/views/wan.css` now includes compact toggle helpers for temporal reset-anchor (`.wan-temporal-anchor-toggle*`) and reflow helpers for upscaling card rows (`.wan-upscaling-*`), keeping toggle buttons content-width and reducing uneven row stretching.
- 2026-02-20: `styles/views/wan.css` adds opt-in clickable `wan-subheader` states (`--clickable` hover/focus) for full-row header toggles; `xyz-sweep-card.css` adds `.xyz-card-body` for collapsed-body layout grouping.
- 2026-02-20: `styles.css` sets `.btn-destructive` height to `2rem` for size parity with adjacent `.btn-outline` controls in compact toolbars.
- 2026-03-02: `generation-settings-card.css` now keeps CFG advanced sub-rows stacked with `cfg-advanced-row--secondary` only (legacy `gc-col--cfg-advanced-apg-eta` removed), matching the three-row Advanced layout used by both basic parameter cards.
- 2026-03-02: `styles/components/hires-settings-card.css` now defines `hr-tile-row`/`hr-tile-col`/`hr-tile-presets`/`hr-tile-slider` hooks to keep Hires Tile presets + Overlap + Min-tile aligned in a single desktop row (with responsive wrap fallback below `72rem`).
- 2026-03-02: `styles/components/hires-settings-card.css` now right-anchors Hires tile preset grid (`.hr-tile-presets { justify-content: flex-end; }`) so tile knobs stay on the right side of the tile row in desktop layout.
- 2026-03-02: `styles/components/refiner-settings-card.css` now uses dedicated row hooks (`rf-row`, `rf-row--advanced`, `rf-row--advanced-secondary`) so swap-model sliders stay on one row and APG advanced rows expand in stable 3-col/2-col layouts (with single-column fallback below `56rem`).
- 2026-02-21: `generation-settings-card.css` now styles an active state for dimension action buttons (`.btn-swap--active`) used by the new aspect-ratio lock toggle in the basic parameter cards.
- 2026-02-20: `styles/components/views-shared.css` now styles History as square 1:1 thumbnail cards (`.cdx-history-item` + `.cdx-history-thumb`, object-fit contain) and adds `cdx-history-modal*` layout rules for organized run-details dialogs.
- 2026-02-21: Added `styles/components/inpaint-mask-editor.css` and imported it in `styles.css` for the new full-screen inpaint mask editor overlay (stage/content/cursor/toolbar semantics).
- 2026-02-21: `img2img-inpaint-params-card.css` now styles inline mask tools anchored under the init-image section (`.img2img-mask-inline-tools`), and `inpaint-mask-editor.css` now hides the internal upload input used by the editor toolbar import action.
- 2026-02-21: `styles/components/result-viewer.css` now constrains image/video previews to `max-height: 30dvh` and proportional viewport width (`max-width: min(100%, 42dvw)`, fallback `100%` on smaller screens), with `object-fit: contain`; `styles/views/wan.css` applies the same height contract to WAN exported-video previews inside Results.
- 2026-03-02: `styles/components/result-viewer.css` now vertically centers the basic zoom toolbar controls (`Fit`, `1:1`, `+`, `-`, `Close`) in `ImageZoomOverlay`, while guide-edit mode (`image-zoom-toolbar--with-guide`) remains top-aligned.
- 2026-03-04: `styles/components/result-viewer.css` now includes dedicated `video-zoom-*` classes used by `VideoZoomOverlay.vue` for full-screen video pan/zoom toolbar layout.
- 2026-03-04: `styles/components/result-viewer.css` now defines explicit `video-zoom-pan-zone*` pointer-event zoning (off by default, active only in `Pan: On`) and removes WAN exported-video center-hitbox trigger styling.
- 2026-02-21: `styles/components/param-blocks.css` now supports `data-cols=\"5\"` grids so WAN `Chunking` can keep five controls in one desktop row while reusing existing shared field layout primitives.
- 2026-02-21: `styles/views/wan.css` now styles `.wan-temporal-controls` / `.wan-temporal-row` so WAN img2vid temporal mode renders as a stable two-row card (row-1 selects, row-2 sliders) without select+slider row-wrap collisions.
- 2026-02-21: Added `styles/components/base-tab-header.css` and moved `BaseTabHeader.vue` action-row alignment/margins out of inline template styles.
- 2026-02-21: `styles.css` modal shell now sizes `.modal-panel` by viewport width (`min(96vw, 72rem)`) instead of `100dvh` width, preserving consistent dialog proportions.
- 2026-02-21: `styles.css` removed duplicate baseline definitions (`.panel-stack`, `.viewer-card`, `.viewer-empty`) and normalized modal list spacing under shared `.modal-list-section`.
- 2026-03-02: `styles.css` now gives `LoraModal` a scrollable list region (`.lora-modal-list-section`/`.lora-modal-list`) with zebra rows + hover emphasis, right-aligned action toggles, and active-toggle visual state (`.lora-modal-action.is-active`).
- 2026-02-21: `styles/components/settings-form.css` now uses namespaced `settings-*` selectors to prevent collisions with global `.form-*` hooks used by other views.
- 2026-02-21: Added `styles/components/bootstrap-screen.css` and `styles/components/dependency-check-panel.css`, migrating remaining scoped style blocks from `App.vue` and `DependencyCheckPanel.vue` into shared style modules.
- 2026-02-21: Removed every `transform: translate...` usage from active UI styles (and the `EXAMPLE-dashboard-surface-base.css` reference sheet), replacing those offsets with `top/left`, `inset`, margin-based centering, and non-translate transitions.
- 2026-02-22: `styles/components/views-shared.css` now defines a richer `run-progress-status` system under shared `panel-status` semantics (variant colors, icon animation, structured header/meta) so Run panels can host progress/error/warning/info/success in one container.
- 2026-02-22: `styles/components/views-shared.css` adds split meta layout helpers (`run-progress-status__meta-left/right`) and `run-progress-status__meta-item--elapsed` so progress panels render elapsed time on the right side opposite Step/ETA.
- 2026-02-23: `styles/components/views-shared.css` adds `.panel-stack--sticky` for generation right-column stickiness (Run + Results follow scroll with `--sticky-offset` on desktop; auto-disabled below the shared one-column breakpoint).
- 2026-03-02: `styles/components/views-shared.css` keeps desktop sticky behavior at the stack level only (`.panel-stack--sticky` with `top: calc(--sticky-offset + space)`), without pinning the first child panel separately.
- 2026-03-02: `styles/components/result-viewer.css` now caps result media to `max-height: 30dvh` with a proportional viewport-width cap (`max-width: min(100%, 42dvw)`, relaxed to `100%` on smaller screens) to avoid oversized previews in Results cards.
- 2026-03-02: `styles/components/views-shared.css` now styles dual run-progress bars (`run-progress-status__bars`, `__bar-group`, `__bar-caption`, `__bar-meta`) with separate visual treatments for `total` and `steps` progress tracks.
- 2026-03-03: `styles/components/quicksettings.css` add-path modal styles (`qs-add-path-*`) include input+scan row, animated scan spinner state, right-aligned add-all action, scrollable zebra candidate table, row status/error states, and modal sizing hooks.
- 2026-03-04: `styles/components/quicksettings.css` add-path modal now styles byte-progress telemetry during add-all (`qs-add-path-progress-caption` + `qs-add-path-progress-bar`) with deterministic numeric alignment and themed progress fills.
- 2026-03-05: `styles/components/quicksettings.css` renamed the Flux dual text-encoder sizing hook from `.qs-group-flux1-tenc` to `.qs-group-flux-tenc` so class naming matches Flux-family scope (`flux1` + `flux2`).
- 2026-02-27: `styles/views/wan.css` now stacks `Ping-pong`/`Return frames` vertically in `WanVideoOutputPanel.vue` (`.wan-video-output-toggle-row` single-column grid), keeping output toggles compact under the Loop/CRF/Interpolation FPS row.
- 2026-03-13: `styles/components/views-shared.css` now owns neutral video result helpers (`results-header-actions`, `results-empty-state`, `results-empty-title`) shared by WAN, LTX, and image-tab result cards; the old WAN-prefixed empty/header-action helpers were removed from `styles/views/wan.css`.
- 2026-03-16: `styles/components/views-shared.css` now owns the shared video-family grid ratio through `.video-panels`, and both WAN/LTX video workspaces use that class; `styles/views/wan.css` no longer owns the canonical two-column video grid.
