# apps/interface/src/views Overview
<!-- tags: frontend, views, txt2img, sdxl -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-05
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
