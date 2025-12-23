<!-- tags: frontend, styles, tokens, conventions -->
# apps/interface/src/styles Overview
Date: 2025-12-22  
Owner: Codex WebUI Frontend  
Last Review: 2025-12-23  
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
