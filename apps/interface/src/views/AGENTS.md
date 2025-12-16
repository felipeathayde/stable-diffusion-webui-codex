# apps/interface/src/views Overview
<!-- tags: frontend, views, txt2img, sdxl -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-16
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
- 2025-12-14: `WANTab.vue` uses typed (Zod) WAN video payload builders and surfaces streaming progress + returned `info` JSON to reduce request drift during WAN22 pipeline testing.
- 2025-12-14: `WANTab.vue` was de-bloated by extracting High/Low + Output panels into `components/wan/*` and treating WAN assets (model dirs/TE/VAE) as QuickSettings responsibility (tab shows a summary + validates before starting runs).
- 2025-12-15: `WANTab.vue` parameters UI was reshaped to mirror Txt2Img’s layout (Prompt + Generation Parameters with cards); prompt now uses `PromptFields`.
- 2025-12-16: `WANTab.vue` adds a `vid2vid` mode (video upload + flow-chunks params) and shows exported video playback when the backend returns `video.rel_path` from `/api/vid2vid`.
- 2025-12-14: Removed the legacy standalone `Txt2Vid.vue` view; WAN video entry stays exclusively under model tabs (`/models/:tabId` with `type === 'wan'`).
- 2025-12-14: `ModelTabView.vue` keys per-tab views by `tab.id` so switching `/models/:tabId` remounts the correct tab implementation (prevents composables binding a stale id).
- 2025-12-15: Added `RedirectToModelTab.vue` and router aliases so legacy nav paths can redirect into `/models/:tabId` (WAN) without spamming Vue Router “No match found” warnings.
- 2025-12-15: Model-tab actions (rename/enable/load/unload/duplicate/remove) were moved out of the per-tab view; `Home.vue` is now the canonical place to manage tabs, and the per-tab “Send to Workflows” action lives in a dedicated panel under Generation Parameters.
