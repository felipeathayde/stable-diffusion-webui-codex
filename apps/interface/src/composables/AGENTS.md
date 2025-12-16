# apps/interface/src/composables Overview
<!-- tags: frontend, composables -->
Date: 2025-12-09
Owner: Frontend Maintainers
Last Review: 2025-12-16
Status: Active

## Purpose
- Vue composables that encapsulate shared generation logic and reusable reactive helpers for engine tabs.

## Notes
- `useGeneration` builds txt2img payloads for model tabs; when an engine declares `requiresTenc`, it calls `quicksettings.resolveTextEncoderSha` and now fails fast with a clear error if no text encoder SHA can be resolved, preventing GGUF runs from hitting backend 400s.
- 2025-12-14: `useVideoGeneration(tabId)` encapsulates WAN `/txt2vid` + `/img2vid` generation + SSE streaming state so `WANTab.vue` stays a thin view.
- 2025-12-16: `useVideoGeneration(tabId)` now supports WAN `vid2vid` via `/api/vid2vid` multipart upload (keeps the selected video file in-memory per tab) and exposes `videoUrl` for exported outputs served by `/api/output/{rel_path}`.
