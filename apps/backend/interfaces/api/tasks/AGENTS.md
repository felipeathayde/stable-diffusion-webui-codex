# apps/backend/interfaces/api/tasks Overview
<!-- tags: backend, api, tasks, orchestration -->
Date: 2026-01-30
Last Review: 2026-02-09
Status: Active

## Purpose
- Host shared, import-light task orchestration helpers used by API routers.
- Keep routers thin: validate + dispatch + stream, while task boilerplate (queue/status/progress/result/end) lives here.

## Key Files
- `apps/backend/interfaces/api/tasks/generation_tasks.py` — common task runner helpers for generation endpoints (event streaming, engine options build, image encoding).
- `apps/backend/interfaces/api/tasks/upscale_tasks.py` — task workers for standalone `/api/upscale` and HF upscaler downloads (`/api/upscalers/download`) with explicit integrity verification (manifest sha256 when available).

## Notes
- This package must remain import-light (avoid importing torch-heavy modules at import time). Prefer local imports inside functions.
- Cancellation semantics are owned by `apps/backend/interfaces/api/task_registry.py` and must be preserved (always emit `end`).
- 2026-02-09: Task workers now compare cancellation policy using `TaskCancelMode.IMMEDIATE` (enumized contract from `task_registry.py`) instead of raw string literals.
