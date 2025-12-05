<!-- tags: frontend, components, prompt, highres, refiner -->
# apps/interface/src/components Overview
Date: 2025-12-05
Owner: Frontend Maintainers
Last Review: 2025-12-05
Status: Active

## Purpose
- Reusable Vue components (panels, form widgets, prompts, galleries) shared across views.

## Notes
- Components should be presentational and rely on Pinia stores or props for state.
- Follow the styling rules documented in `.sangoi/frontend/guidelines/frontend-style-guide.md`.
- Prompt parsing/serialization lives in `prompt/PromptToken.ts` with Vitest coverage; ensure new prompt widgets pass through that module.
- Generation + highres + refiner controls live in `GenerationSettingsCard.vue`, `HighresSettingsCard.vue`, and `RefinerSettingsCard.vue`, all using CSS grid layouts.
- Model/sampler/scheduler dropdowns vivem em `ModelSelector.vue`, `SamplerSelector.vue` e `SchedulerSelector.vue`; presets usam `PresetsSelector.vue` com datalist + Apply; views como `/flux` devem reutilizar esses componentes em vez de construir selects ad-hoc.
- `QuickSettingsBar.vue` surfaces global engine/preset controls e um bloco compacto de “Per-component overrides” para core/TE/VAE device/dtype; o bloco de overrides continua avançado/colapsável, enquanto toggles como Smart Offload/Smart Fallback aparecem na faixa principal para espelhar o mental model Forge/A1111 sem poluir o layout.
- `ResultViewer.vue` exibe um overlay full-screen para zoom de imagens (sem modal encaixotado): o preview da galeria continua grande no card, enquanto o overlay usa o viewport inteiro com ferramenta lateral para pan/zoom (drag para pan, botões de Fit/1:1/+/−/Close na barra à direita).
