# apps/backend/runtime/workflows Overview
Date: 2025-10-30
Owner: Runtime Maintainers
Last Review: 2026-01-03
Status: Active

## Purpose
- Provide shared orchestration helpers for Codex generation workflows (text, image, and video).
- Centralize stage logic (prompt normalization, sampler planning, init-image prep, video metadata) to keep use cases lightweight and consistent.

## Key Files
- `prompt_context.py` — Prompt parsing + prompt-derived controls (clip_skip, width/height overrides).
- `sampling_plan.py` — Scheduler normalization, noise settings, plan building, sampler/RNG preparation.
- `sampling_execute.py` — Sampler execution + live preview callback + latent dump diagnostics.
- `scripts.py` — Script hooks + extra network (LoRA) activation helpers.
- `image_io.py` — PIL/tensor conversions + optional hires decode helper.
- `tiling.py` — VAE tiling apply/restore toggles.
- `image_init.py` — Utilities for encoding img2img/img2vid init images into tensor+latent bundles.
- `video.py` — Video plan builder, LoRA application, sampler configuration, and metadata assembly.
- `__init__.py` — Package marker (intentionally no re-export facade; callers import modules directly).

## Notes
- Modules in this directory must stay dependency-light and only import from `apps.*` namespaces.
- Prefer adding new workflow stages here rather than duplicating logic inside `apps/backend/use_cases/`.
- Helper functions should raise explicit errors; avoid silent fallbacks or catching broad exceptions.
- 2025-12-14: `build_video_plan()` reads `steps` + `guidance_scale` directly from the request; LoRA application uses a lazy import to keep the module dependency-light for non-LoRA users.
- 2025-12-14: Video plan defaults `steps` to 30 when an ad-hoc caller omits it (matching `/api/{txt2vid,img2vid}` defaults) to avoid drifting configs.
- 2026-01-01: `clip_skip` is now treated as a prompt control applied in `apply_prompt_context(...)` (before conditioning is computed); request-level `clip_skip` is merged into `PromptContext.controls` when no `<clip_skip:…>` tag is present.
- 2026-01-01: The live preview callback in `common.py` uses `runtime/live_preview.py` (method enum + decode helper) and records the sampling step in `backend_state` when emitting preview images.
- 2026-01-01: Preview-factor fitting/logging (least-squares latent→RGB `factors`/`bias`) lives in `runtime/live_preview.py` and can be enabled via `CODEX_DEBUG_PREVIEW_FACTORS=1` (used to derive `Approx cheap` mappings for new latent formats).
- 2026-01-02: Removed token merging application from `common.py`; `<merge:...>` / `<tm:...>` tags are stripped but have no effect.
- 2026-01-02: Added standardized file header docstrings to workflow helper modules (doc-only change; part of rollout).
- 2026-01-03: Split the former `common.py` golema into focused modules and removed the workflow re-export facade (`__init__.py` is now intentionally empty).
