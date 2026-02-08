<!-- tags: frontend, styles, tokens, conventions -->
# apps/interface/src/styles Overview
Date: 2025-12-22  
Last Review: 2026-02-08  
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
  - View-specific styling/layout (WAN, Settings, XYZ, etc.).
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
- `cd apps/interface && npm test`

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
- 2025-12-27: Added shared helpers for input-inline actions and chip rows (`cdx-input-with-actions`, `cdx-chips-row`) and a `cdx-res-presets` block for aligning size presets with Width/Height (2×2 grid).
- 2025-12-27: Prompt toolbar Styles field now has a min width to prevent the inline actions from overlapping the input text.
- 2025-12-26: Standardized `.panel-header` height to `4.25rem` (min height) and removed title-wrapper `<span>` patterns from panel headers.
- 2025-12-26: QuickSettings buttons now use `qs-btn-secondary`/`qs-btn-outline` so they fill the `qs-row` height with consistent borders (no fixed `2rem` height).
- 2025-12-28: QuickSettings toggles now use `qs-toggle-btn` (neon border states) and the legacy `qs-switch` styling was removed.
- 2026-02-08: `quicksettings.css` now styles a paired mode-toggle group (`IMG2IMG`/`INPAINT`) in the top row and applies explicit disabled-state treatment for `qs-toggle-btn:disabled`.
- 2025-12-28: QuickSettings bar now stacks `quicksettings-row` blocks and uses an animated (rAF) collapsible Advanced row; WAN adds `wan-subheader` section headers in `styles/views/wan.css`.
- 2025-12-29: Sticky header offset (`--sticky-offset`) is derived from the `.main-header` height and used by `RunCard` (`.panel-header.results-sticky`).
- 2026-01-01: History cards now render as a single-row horizontal strip (with horizontal scroll); per-item action buttons appear on hover as a compact overlay (with horizontal scroll when needed) and stay visible for the selected item via `styles/components/views-shared.css` (image tabs + WAN).
- 2026-01-14: Tools view now right-aligns GGUF action buttons (Overwrite/Comfy Layout + Convert) via `styles/views/tools.css`.
- 2026-01-29: Added `styles/components/cdx-dropzone.css` and `styles/views/pnginfo.css` for the revamped PNG Info view/dropzone.
- 2026-02-03: Renamed the hires settings card stylesheet to `styles/components/hires-settings-card.css` and updated the card root class to `.hires-card`.
