# apps/backend/runtime/workflows Overview
Date: 2025-10-30
Owner: Runtime Maintainers
Last Review: 2025-10-30
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
