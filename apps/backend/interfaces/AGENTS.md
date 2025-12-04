# apps/backend/interfaces Overview
<!-- tags: backend, api, validation -->
Date: 2025-10-28
Owner: Backend API Maintainers
Last Review: 2025-12-03
Status: Active

## Purpose
- Defines API-facing schemas and adapters that expose backend capabilities to the Codex frontend and external clients.

## Subdirectories
- `api/` — FastAPI endpoint implementations and adapters.
- `schemas/` — Pydantic/dataclass schemas describing request/response payloads.

## Notes
- Keep schemas in sync with the frontend API client (`apps/interface/src/api`).
- Avoid embedding business logic here—delegate to services/use cases and focus on validation and serialization.
- API workers should reuse a single `InferenceOrchestrator` instance per process to preserve engine caches/VRAM across requests. See `api/run_api.py` (`_ORCH` singleton).
- 2025-11-14: `/api/txt2img` enforces the semantic contract (e.g., `prompt`, `negative_prompt`, `width`, `extras.highres`) but still tolerates compatibility keys (`codex_engine`, `codex_diffusion_device`, `sd_model_checkpoint`) while downstream clients migrate; prompts may be empty to support negative-only runs.
- 2025-11-21: SPA static mount now registers after all `/api/*` routes to prevent POSTs from being intercepted by the UI fallback; invalid txt2/img2/video payloads raise HTTP errors instead of returning 200 with a background error.
- 2025-11-21: Module-level `app` remains available for ASGI servers, but the preferred entrypoint is the uvicorn factory `apps.backend.interfaces.api.run_api:create_api_app`. Factory and direct `:app` both build the same FastAPI instance.
- 2025-11-14: `create_api_app(argv, env)` is the canonical FastAPI factory; when launching uvicorn manually use `--factory apps.backend.interfaces.api.run_api:create_api_app` so the runtime bootstraps before serving (the TUI/launcher already calls it).
- 2025-12-03: `/api/txt2img` extras now accept `highres.refiner` (enable/steps/cfg/seed/model/vae) alongside the global `extras.refiner`, raising HTTP 400 on malformed nested refiner configs.
- 2025-12-03: `/api/tasks/{task_id}/cancel` allows best-effort cancellation (immediate vs after_current flag); workers abort event streaming with `error: cancelled` when `mode=immediate`.
- 2025-12-03: `/api/options` now accepts `codex_{core,te,vae}_{device,dtype}` to set per-role backend/dtype via memory manager; device choices auto/cuda/cpu/mps/xpu/directml, dtype auto/fp16/bf16/fp32.
