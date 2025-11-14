# apps/backend/interfaces Overview
Date: 2025-10-28
Owner: Backend API Maintainers
Last Review: 2025-10-28
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
- 2025-11-14: `create_api_app(argv, env)` is the canonical FastAPI factory; when launching uvicorn manually use `--factory apps.backend.interfaces.api.run_api:create_api_app` so the runtime bootstraps before serving (the TUI/launcher already calls it).
