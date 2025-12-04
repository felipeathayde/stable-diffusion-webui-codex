<!-- tags: frontend, components, prompt, highres, refiner -->
# apps/interface/src/components Overview
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-04
Status: Active

## Purpose
- Reusable Vue components (panels, form widgets, prompts, galleries) shared across views.

## Notes
- Components should be presentational and rely on Pinia stores or props for state.
- Follow the styling rules documented in `.sangoi/frontend/guidelines/frontend-style-guide.md`.
- Prompt parsing/serialization lives in `prompt/PromptToken.ts` with Vitest coverage; ensure new prompt widgets pass through that module.
- Generation + highres + refiner controls live in `GenerationSettingsCard.vue`, `HighresSettingsCard.vue`, and `RefinerSettingsCard.vue`, all using CSS grid layouts.
- `QuickSettingsBar.vue` surfaces global engine/preset controls and a compact “Per-component overrides” section for core/TE/VAE device/dtype; the overrides block is collapsible and should remain advanced-only so the default header stays close to the Forge/A1111 quicksettings mental model.
