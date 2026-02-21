# apps/interface/src/composables Overview
<!-- tags: frontend, composables -->
Date: 2025-12-09
Last Review: 2026-02-21
Status: Active

## Purpose
- Vue composables that encapsulate shared generation logic and reusable reactive helpers for engine tabs.

## Notes
- `useGeneration(tabId)` builds txt2img payloads for model tabs using tab-scoped selections (`tab.params.checkpoint`, `tab.params.textEncoders`). It fails fast when no checkpoint is selected and when required `tenc_sha` values can’t be resolved (engine requires TE, including Flux/ZImage, or a GGUF checkpoint is selected).
- 2025-12-28: `useGeneration(tabId)` now propagates tab-scoped `batchCount`/`batchSize` into txt2img/img2img payloads (previously fixed to 1×1) and tracks `progress`/`info`/`gentimeMs` for the image-tabs Results UI.
- 2025-12-28: `useGeneration(tabId)` now maintains a small per-tab image run history (task id + params snapshot) and exposes `loadHistory/clearHistory` so views can render a History panel.
- 2025-12-31: FLUX.1 img2img requests from `useGeneration(tabId)` are routed to the Kontext workflow engine (canonical key `engine="flux1_kontext"`, previously `kontext`) and include `img2img_extras.tenc_sha` (no `text_encoder_override` for Flux.1; backend derives the override from sha).
- 2026-01-29: `useGeneration(tabId)` now emits Codex-native masked img2img (“inpaint”) fields when `tab.params.useMask` is enabled (`img2img_mask_enforcement`, full-res crop/padding, invert/round/blur, masked-content mode). Flux.1 (Kontext) masking is blocked explicitly until semantics are implemented.
- 2025-12-14: `useVideoGeneration(tabId)` encapsulates WAN `/txt2vid` + `/img2vid` generation + SSE streaming state so `WANTab.vue` stays a thin view.
- 2025-12-16: `useVideoGeneration(tabId)` exposes WAN video task execution + export URL wiring (`/api/output/{rel_path}`) for the active WAN modes.
- 2026-01-17: `useVideoGeneration(tabId)` now derives `wan_metadata_repo` for the known WAN2.2 repos from the current input mode + size hint (prevents stale repo ids after switching txt/img/vid modes).
- 2026-01-21: `useVideoGeneration(tabId)` builds WAN stage payloads with sha-only LoRA inputs (`loraSha` → `lora_sha`).
- 2026-01-27: `useVideoGeneration(tabId)` propagates the WAN `Return frames` toggle via `video_return_frames` (default off); frames are still returned when `Save output` is off or export fails (backend attaches `info.warnings`).
- 2025-12-25: `usePromptCard` encapsulates shared prompt toolbar state (TI/LoRA/Styles) and negative-prompt visibility defaults.
- 2025-12-27: Removed Checkpoints state from `usePromptCard` (PromptCard no longer renders the Checkpoints modal/button).
- 2025-12-25: `useResultsCard` encapsulates shared “Results” helpers (clipboard copy + ephemeral notice/toast + JSON formatting) so views don’t duplicate the same wiring.
- 2025-12-27: Added `useModelTabNavigation` to bridge “Send to Img2Img/Inpaint” actions into `/models/:tabId` tabs by setting init-image params.
- 2026-01-01: `useGeneration(tabId)` now tracks live preview images from task progress events (`previewImage`/`previewStep`) and sets the initial stage to `starting` immediately on Generate click (so Results doesn’t read as “No results yet” during request setup).
- 2026-01-02: `useGeneration(tabId)` now resolves the selected checkpoint to its short hash (when available) before sending requests.
- 2026-01-18: `useGeneration(tabId)` now derives required VAE/text encoder count from backend-provided `asset_contracts` and uses `models[].core_only` (via `quicksettings.isModelCoreOnly(...)`) to enforce core-only requirements (no duplicated per-engine lists in the UI).
- 2026-01-18: `useGeneration(tabId)` maps the `chroma` tab type to backend engine id `flux1_chroma` (keeps tab taxonomy explicit while requests use canonical engine keys).
- 2026-01-28: `useGeneration(tabId)` emits `extras.zimage_variant="turbo"|"base"` for Z-Image requests; both variants use classic CFG (negative prompts supported) and the toggle exists to drive scheduler shift + recommended defaults.
- 2026-02-03: `useGeneration(tabId)` hires payloads emit `extras.hires` in txt2img requests.
- 2026-02-04: `useGeneration(tabId)` propagates the global `min_tile` preference into txt2img hires tile payloads (`extras.hires.tile.min_tile`, clamped to `tile`).
- 2026-01-03: Added standardized file header blocks to composables (doc-only change; part of rollout).
- 2026-02-05: `useGeneration(tabId)` now hard-checks backend engine surface before request send; missing capabilities or unsupported mode (`supports_txt2img`/`supports_img2img`) fail loud with explicit errors.
- 2026-02-05: `useGeneration` now exports `resolveEngineForRequest(...)` as the canonical tab-type/mode engine mapper; `ImageModelTab.vue` reuses it so disable-state and request preflight stay in parity.
- 2026-02-06: `useVideoGeneration(tabId)` default WAN video params now include `returnFrames` (default false) and align interpolation defaults (`rifeEnabled`/`rifeModel`) with the canonical WAN params surface (prevents drift between store/view/composable defaults).
- 2026-02-06: `useGeneration` engine mapping now delegates to `utils/engine_taxonomy.ts` so request engine-id resolution (`flux1_kontext`, `flux1_chroma`) is centralized and shared with other frontend modules.
- 2026-02-08: `useGeneration.ts` now exports `isGenerationRunningForTab(tabId)` so header quicksettings controls can enforce run-lock behavior on mode toggles (e.g., INPAINT).
- 2026-02-18: `useGeneration.ts` keeps img2img payloads hires-free at the payload source (no `img2img_hires_*` keys are emitted); `img2imgResizeMode`/`img2imgUpscaler` are UI-state fields only (layout/selection, no hires dispatch).
- 2026-02-18: `useGeneration.ts` now builds optional `extras.guidance` / `img2img_extras.guidance` from `tab.params.guidanceAdvanced`, gated by backend `engineSurface.guidance_advanced` so unsupported engines/controls are omitted at source.
- 2026-02-15: `useGeneration(tabId)` and `useVideoGeneration(tabId)` now emit `settings_revision` on every start payload and handle stale-revision backend conflicts (`409` + `current_revision`) by refreshing revision state and surfacing a manual-retry message.
- 2026-02-15: Added `settings_revision_conflict.ts` shared composable helper for parsing/formatting stale-settings conflict UX across image/video generation.
- 2026-02-06: `useVideoGeneration(tabId)` now consumes typed WAN tab params (`TabByType<'wan'>`) and shared `WanAssetsParams` from `model_tabs.ts` instead of local `tab.params as any` casting for core WAN param reads.
- 2026-02-06: `useVideoGeneration(tabId)` now imports shared `WanAssetsParams` from `model_tabs.ts` (no local duplicate interface), keeping WAN asset typing aligned across store/view/composable layers.
- 2026-02-16: `useVideoGeneration(tabId)` now propagates `output.returnFrames` into common WAN payload input for all modes, fixing dropped `video_return_frames` requests from the composable path.
- 2026-02-16: `useVideoGeneration(tabId)` now propagates WAN stage `flowShift` (`high/low.flowShift` → `wan_high/wan_low.flow_shift`) so distill runs can enforce non-default scheduler shifts explicitly.
- 2026-02-17: `useVideoGeneration(tabId)` now propagates WAN attention mode (`global|sliding`) and img2vid chunk controls (chunk/overlap/anchor/seed mode) into payload builders and run-history snapshots.
- 2026-02-20: `useGeneration(tabId)` now fails loud on empty VAE selection before payload submission by using `quicksettings.requireVaeSelection()` (prevents blank VAE submits on SDXL/related tabs).
- 2026-02-20: `useGeneration(tabId)` and `useVideoGeneration(tabId)` history entries now include optional `thumbnail` previews (`GeneratedImage`) updated during progress/result flow, enabling square thumbnail-only History cards with detail modal drill-down in views.
- 2026-02-21: `useGeneration(tabId)` now fails loud when a non-sentinel VAE label cannot be resolved to `vae_sha` (`Selected VAE is invalid or stale`), preventing stale hidden selections from degrading into implicit built-in behavior.
- 2026-02-21: `useVideoGeneration(tabId)` now treats WAN prompts as stage-owned (`high.prompt/negativePrompt`, `low.prompt/negativePrompt`), blocks generation if either stage prompt is empty, snapshots both stage prompts in history, and keeps top-level API prompt compatibility by deriving mode prompt from the High stage in payload builders.
- 2026-02-21: `useVideoGeneration(tabId)` now dispatches only `txt2vid|img2vid`; frontend `vid2vid` run preparation/dispatch and init-video file state were removed to match current backend contract exposure.
- 2026-02-21: `useVideoGeneration(tabId)` resume-state parsing is now strict for mode (`txt2vid|img2vid` only): unsupported legacy values (e.g. `vid2vid`) are rejected fail-loud, the stale resume marker is cleared, and UI surfaces a clear resume notice instead of silently downgrading mode.
