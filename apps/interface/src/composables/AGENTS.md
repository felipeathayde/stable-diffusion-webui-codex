# apps/interface/src/composables Overview
<!-- tags: frontend, composables -->
Date: 2025-12-09
Owner: Frontend Maintainers
Last Review: 2025-12-28
Status: Active

## Purpose
- Vue composables that encapsulate shared generation logic and reusable reactive helpers for engine tabs.

## Notes
- `useGeneration(tabId)` builds txt2img payloads for model tabs using tab-scoped selections (`tab.params.checkpoint`, `tab.params.textEncoders`). It fails fast when no checkpoint is selected and when required `tenc_sha` values can’t be resolved (engine requires TE or a GGUF checkpoint is selected), preventing backend 400s; Flux also derives `extras.text_encoder_override` from `textEncoders` when present.
- 2025-12-28: `useGeneration(tabId)` now propagates tab-scoped `batchCount`/`batchSize` into txt2img/img2img payloads (previously fixed to 1×1) and tracks `progress`/`info`/`gentimeMs` for the image-tabs Results UI.
- 2025-12-14: `useVideoGeneration(tabId)` encapsulates WAN `/txt2vid` + `/img2vid` generation + SSE streaming state so `WANTab.vue` stays a thin view.
- 2025-12-16: `useVideoGeneration(tabId)` now supports WAN `vid2vid` via `/api/vid2vid` multipart upload (keeps the selected video file in-memory per tab) and exposes `videoUrl` for exported outputs served by `/api/output/{rel_path}`.
- 2025-12-25: `usePromptCard` encapsulates shared prompt toolbar state (TI/LoRA/Styles) and negative-prompt visibility defaults.
- 2025-12-27: Removed Checkpoints state from `usePromptCard` (PromptCard no longer renders the Checkpoints modal/button).
- 2025-12-25: `useResultsCard` encapsulates shared “Results” helpers (clipboard copy + ephemeral notice/toast + JSON formatting) so views don’t duplicate the same wiring.
- 2025-12-27: Added `useModelTabNavigation` to bridge “Send to Img2Img/Inpaint” actions into `/models/:tabId` tabs by setting init-image params.
