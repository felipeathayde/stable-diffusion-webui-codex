# apps/interface/src/composables Overview
<!-- tags: frontend, composables -->
Date: 2025-12-09
Owner: Frontend Maintainers
Last Review: 2026-01-27
Status: Active

## Purpose
- Vue composables that encapsulate shared generation logic and reusable reactive helpers for engine tabs.

## Notes
- `useGeneration(tabId)` builds txt2img payloads for model tabs using tab-scoped selections (`tab.params.checkpoint`, `tab.params.textEncoders`). It fails fast when no checkpoint is selected and when required `tenc_sha` values can’t be resolved (engine requires TE, including Flux/ZImage, or a GGUF checkpoint is selected).
- 2025-12-28: `useGeneration(tabId)` now propagates tab-scoped `batchCount`/`batchSize` into txt2img/img2img payloads (previously fixed to 1×1) and tracks `progress`/`info`/`gentimeMs` for the image-tabs Results UI.
- 2025-12-28: `useGeneration(tabId)` now maintains a small per-tab image run history (task id + params snapshot) and exposes `loadHistory/clearHistory` so views can render a History panel.
- 2025-12-31: FLUX.1 img2img requests from `useGeneration(tabId)` are routed to the Kontext workflow engine (canonical key `engine="flux1_kontext"`, previously `kontext`) and include `img2img_extras.tenc_sha` (no `text_encoder_override` for Flux.1; backend derives the override from sha).
- 2025-12-14: `useVideoGeneration(tabId)` encapsulates WAN `/txt2vid` + `/img2vid` generation + SSE streaming state so `WANTab.vue` stays a thin view.
- 2025-12-16: `useVideoGeneration(tabId)` now supports WAN `vid2vid` via `/api/vid2vid` multipart upload (keeps the selected video file in-memory per tab) and exposes `videoUrl` for exported outputs served by `/api/output/{rel_path}`.
- 2026-01-17: `useVideoGeneration(tabId)` now derives `wan_metadata_repo` for the known WAN2.2 repos from the current input mode + size hint (prevents stale repo ids after switching txt/img/vid modes).
- 2026-01-21: `useVideoGeneration(tabId)` builds WAN stage payloads with sha-only LoRA inputs (`loraSha` → `lora_sha`).
- 2026-01-27: `useVideoGeneration(tabId)` propagates the WAN `Return frames` toggle via `video_return_frames` (default off); frames are still returned when `Save output` is off or export fails (backend attaches `info.warnings`). Vid2vid uses the same toggle to control preview frames.
- 2025-12-25: `usePromptCard` encapsulates shared prompt toolbar state (TI/LoRA/Styles) and negative-prompt visibility defaults.
- 2025-12-27: Removed Checkpoints state from `usePromptCard` (PromptCard no longer renders the Checkpoints modal/button).
- 2025-12-25: `useResultsCard` encapsulates shared “Results” helpers (clipboard copy + ephemeral notice/toast + JSON formatting) so views don’t duplicate the same wiring.
- 2025-12-27: Added `useModelTabNavigation` to bridge “Send to Img2Img/Inpaint” actions into `/models/:tabId` tabs by setting init-image params.
- 2026-01-01: `useGeneration(tabId)` now tracks live preview images from task progress events (`previewImage`/`previewStep`) and sets the initial stage to `starting` immediately on Generate click (so Results doesn’t read as “No results yet” during request setup).
- 2026-01-02: `useGeneration(tabId)` now resolves the selected checkpoint to its short hash (when available) before sending requests.
- 2026-01-18: `useGeneration(tabId)` now derives required VAE/text encoder count from backend-provided `asset_contracts` and uses `models[].core_only` (via `quicksettings.isModelCoreOnly(...)`) to enforce core-only requirements (no duplicated per-engine lists in the UI).
- 2026-01-18: `useGeneration(tabId)` maps the `chroma` tab type to backend engine id `flux1_chroma` (keeps tab taxonomy explicit while requests use canonical engine keys).
- 2026-01-03: Added standardized file header blocks to composables (doc-only change; part of rollout).
