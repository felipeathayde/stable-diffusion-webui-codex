# apps/backend/interfaces/api/routers Overview
<!-- tags: backend, api, fastapi, routers -->
Date: 2026-01-08
Owner: Backend API Maintainers
Last Review: 2026-01-08
Status: Active

## Purpose
- Group FastAPI routes by responsibility; each module exposes a `build_router(...)` factory.

## Modules
- `apps/backend/interfaces/api/routers/system.py` — health/version/memory endpoints.
- `apps/backend/interfaces/api/routers/settings.py` — settings schema + values endpoints.
- `apps/backend/interfaces/api/routers/ui.py` — tabs/workflows/blocks/presets persistence endpoints.
- `apps/backend/interfaces/api/routers/models.py` — model inventory + samplers/schedulers/VAEs/encoders/LoRA endpoints.
- `apps/backend/interfaces/api/routers/paths.py` — `apps/paths.json` endpoints.
- `apps/backend/interfaces/api/routers/options.py` — options store read/update/validate endpoints.
- `apps/backend/interfaces/api/routers/tasks.py` — task status/SSE/output endpoints.
- `apps/backend/interfaces/api/routers/tools.py` — GGUF converter + file browser endpoints.
- `apps/backend/interfaces/api/routers/generation.py` — txt2img/img2img/txt2vid/img2vid/vid2vid endpoints.

## Notes
- Routers should not mutate global state in `run_api.py`; prefer explicit dependency injection via `build_router(...)`.
