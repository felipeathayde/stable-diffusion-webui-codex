<!-- tags: frontend, components, results -->
# apps/interface/src/components/results Overview
Date: 2025-12-25
Last Review: 2026-02-17
Status: Active

## Purpose
- Shared "Results" panel components for generation views (header layout + action slots).

## Notes
- `ResultsCard.vue` provides the standard 3-column Results header (title / center / right); sticky behavior is enabled by including `results-sticky` in `headerClass`. It also exposes `bodyStyle/bodyClass` passthrough for views that need dynamic preview sizing.
- `RunCard.vue` provides the “Run” header (sticky Generate, optional batch dropdown for count/size, optional header-right actions slot) that sits above Results in generation views; batch changes apply immediately (OK closes the dropdown).
- `RunSummaryChips.vue` renders a compact “run summary” string as chips for the Run body (mirrors the prior `caption` summary line, but more scannable).
- `RunProgressStatus.vue` is the canonical Stage/Progress/Step/ETA block used in Run cards across image, WAN, and upscale views.
- 2026-01-02: Added standardized file header docstring to `RunCard.vue` (doc-only change; part of rollout).
- 2026-01-03: Added standardized file header blocks to Results components (doc-only change; part of rollout).
