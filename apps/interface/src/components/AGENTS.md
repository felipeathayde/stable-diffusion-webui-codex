<!-- tags: frontend, components, prompt, highres, refiner -->
# apps/interface/src/components Overview
Date: 2025-12-06
Owner: Frontend Maintainers
Last Review: 2025-12-06
Status: Active

## Purpose
- Reusable Vue components (panels, form widgets, prompts, galleries) shared across views.

## Notes
- Components should be presentational and rely on Pinia stores or props for state.
- Follow the styling rules documented in `.sangoi/frontend/guidelines/frontend-style-guide.md`.
- Prompt parsing/serialization lives in `prompt/PromptToken.ts` with Vitest coverage; ensure new prompt widgets pass through that module.
- Generation + highres + refiner controls live in `GenerationSettingsCard.vue`, `HighresSettingsCard.vue`, and `RefinerSettingsCard.vue`, all using CSS grid layouts.
- Model/sampler/scheduler dropdowns vivem em `ModelSelector.vue`, `SamplerSelector.vue` e `SchedulerSelector.vue`; views devem reutilizar esses componentes em vez de construir selects ad-hoc. Presets/estilos são tratados hoje pelas próprias views (SDXL/Flux) sem um componente dedicado de selector.
- `QuickSettingsBar.vue` surfaces global engine/preset controls e um bloco compacto de “Per-component overrides” para core/TE/VAE device/dtype; o header usa um grid de duas linhas (linha 1: modo/checkpoint/VAE/text encoder/refresh de modelos; linha 2: attention backend, overrides e controles de performance), e continua detectando `/sdxl` e `/flux` mesmo sem model tab ativo para filtrar checkpoints/VAEs/text encoders por família.
- `ResultViewer.vue` exibe um overlay full-screen para zoom de imagens (sem modal encaixotado): o preview da galeria continua grande no card, enquanto o overlay usa o viewport inteiro com ferramenta lateral para pan/zoom (drag para pan, botões de Fit/1:1/+/−/Close na barra à direita).
