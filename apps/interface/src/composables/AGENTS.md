# apps/interface/src/composables Overview
<!-- tags: frontend, composables -->
Date: 2025-12-09
Owner: Frontend Maintainers
Last Review: 2025-12-09
Status: Active

## Purpose
- Vue composables that encapsulate shared generation logic and reusable reactive helpers for engine tabs.

## Notes
- `useGeneration` builds txt2img payloads for model tabs; when an engine declares `requiresTenc`, it calls `quicksettings.resolveTextEncoderSha` and now fails fast with a clear error if no text encoder SHA can be resolved, preventing GGUF runs from hitting backend 400s.
