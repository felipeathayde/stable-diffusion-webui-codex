# apps/backend/runtime/workflows Overview
Date: 2025-10-30
Owner: Runtime Maintainers
Last Review: 2026-01-01
Status: Active

## Purpose
- Provide shared orchestration helpers for Codex generation workflows (text, image, and video).
- Centralize stage logic (prompt normalization, sampler planning, init-image prep, video metadata) to keep use cases lightweight and consistent.

## Key Files
- `common.py` — Prompt parsing, sampling plan helpers, sampler execution, tiling toggles.
- `image_init.py` — Utilities for encoding img2img/img2vid init images into tensor+latent bundles.
- `video.py` — Video plan builder, LoRA application, sampler configuration, and metadata assembly.
- `__init__.py` — Re-exports workflow helpers for downstream modules.

## Notes
- Modules in this directory must stay dependency-light and only import from `apps.*` namespaces.
- Prefer adding new workflow stages here rather than duplicating logic inside `apps/backend/use_cases/`.
- Helper functions should raise explicit errors; avoid silent fallbacks or catching broad exceptions.
- 2025-12-14: `build_video_plan()` reads `steps` + `guidance_scale` directly from the request; LoRA application uses a lazy import to keep the module dependency-light for non-LoRA users.
- 2025-12-14: Video plan defaults `steps` to 30 when an ad-hoc caller omits it (matching `/api/{txt2vid,img2vid}` defaults) to avoid drifting configs.
- 2026-01-01: `clip_skip` is now treated as a prompt control applied in `apply_prompt_context(...)` (before conditioning is computed); request-level `clip_skip` is merged into `PromptContext.controls` when no `<clip_skip:…>` tag is present.
- 2026-01-01: The live preview callback in `common.py` uses `runtime/live_preview.py` (method enum + decode helper) and records the sampling step in `backend_state` when emitting preview images.
- 2026-01-01: Preview-factor fitting/logging (least-squares latent→RGB `factors`/`bias`) lives in `runtime/live_preview.py` and can be enabled via `CODEX_DEBUG_PREVIEW_FACTORS=1` (used to derive `Approx cheap` mappings for new latent formats).
