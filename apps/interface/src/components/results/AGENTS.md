<!-- tags: frontend, components, results -->
# apps/interface/src/components/results Overview
Date: 2025-12-25
Last Review: 2026-02-22
Status: Active

## Purpose
- Shared "Results" panel components for generation views (header layout + action slots).

## Notes
- `ResultsCard.vue` provides the standard 3-column Results header (title / center / right); sticky behavior is enabled by including `results-sticky` in `headerClass`. It also exposes `bodyStyle/bodyClass` passthrough for views that need dynamic preview sizing.
- `RunCard.vue` provides the “Run” header (Generate CTA, optional batch dropdown for count/size, optional header-right actions slot) that sits above Results in generation views; batch changes apply immediately (OK closes the dropdown). While a run is active, the center CTA switches to a destructive two-click cancel confirm (`Cancel` -> `Are you sure?`, 4s timeout).
- `RunSummaryChips.vue` renders a compact “run summary” string as chips for the Run body (mirrors the prior `caption` summary line, but more scannable).
- `RunProgressStatus.vue` is the canonical run-status panel used in Run cards across image, WAN, and upscale views, including severity variants (`progress|error|warning|info|success`), color semantics, animated SVG icons, and a right-aligned elapsed timer on progress rows.
- 2026-01-02: Added standardized file header docstring to `RunCard.vue` (doc-only change; part of rollout).
- 2026-01-03: Added standardized file header blocks to Results components (doc-only change; part of rollout).
- 2026-02-22: `RunProgressStatus.vue` expanded from a progress-only block into a unified status surface; callers now route run errors/notices here instead of Prompt/caption-local panels.
- 2026-02-22: `RunProgressStatus.vue` progress meta row now includes a right-side elapsed timer (`Elapsed mm:ss`/`hh:mm:ss`) opposite Step/ETA metadata.
- 2026-02-22: `RunCard.vue` now owns the run-cancel UX in the center CTA (destructive two-click confirm with a 4s reset window), replacing per-view header cancel buttons.
- 2026-03-02: `RunProgressStatus.vue` now renders dual progress bars for generation runs (upper `total` bar with phase label + lower `steps` bar), while preserving existing status variants and elapsed/ETA metadata layout.
