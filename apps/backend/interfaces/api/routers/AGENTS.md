# apps/backend/interfaces/api/routers Overview
<!-- tags: backend, api, fastapi, routers -->
Date: 2026-01-08
Owner: Backend API Maintainers
Last Review: 2026-01-14
Status: Active

## Purpose
- Group FastAPI routes by responsibility; each module exposes a `build_router(...)` factory.

## Modules
- `apps/backend/interfaces/api/routers/system.py` — health/version/memory endpoints.
- `apps/backend/interfaces/api/routers/settings.py` — settings schema + values endpoints.
- `apps/backend/interfaces/api/routers/ui.py` — tabs/workflows/blocks/presets persistence endpoints.
- `apps/backend/interfaces/api/routers/models.py` — model inventory + samplers/schedulers/embeddings + engine capabilities endpoints.
- `apps/backend/interfaces/api/routers/paths.py` — `apps/paths.json` endpoints.
- `apps/backend/interfaces/api/routers/options.py` — options store read/update/validate endpoints.
- `apps/backend/interfaces/api/routers/tasks.py` — task status/SSE/output endpoints.
- `apps/backend/interfaces/api/routers/tools.py` — GGUF converter + file browser endpoints.
- `apps/backend/interfaces/api/routers/generation.py` — txt2img/img2img/txt2vid/img2vid/vid2vid endpoints.
- `apps/backend/interfaces/api/routers/models.py` also exposes `/api/models/checkpoint-metadata` (UI metadata modal payload for a checkpoint selection).

## Notes
- Routers should not mutate global state in `run_api.py`; prefer explicit dependency injection via `build_router(...)`.
- 2026-01-13: `tools.py` supports GGUF conversion cancellation (`POST /api/tools/convert-gguf/:job_id/cancel`) and an `overwrite` flag (default false; fails with 409 if the output path exists).
- 2026-01-14: `tools.py` accepts a `comfy_layout` flag for GGUF conversion to control Flux/ZImage Comfy/Codex remapping (default true).
- 2026-01-13: `models.py` adds `/api/models/checkpoint-metadata` so the UI can fetch the full metadata modal payload without constructing it client-side.
