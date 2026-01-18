# apps/backend/interfaces/api Overview
<!-- tags: backend, api, fastapi, routers -->
Date: 2026-01-08
Owner: Backend API Maintainers
Last Review: 2026-01-18
Status: Active

## Purpose
- FastAPI endpoint implementations and helper modules for the Codex backend API.

## Key Files
- `apps/backend/interfaces/api/run_api.py` — uvicorn factory + FastAPI assembly (router composition + SPA mount).
- `apps/backend/interfaces/api/routers/system.py` — health/version/memory endpoints.
- `apps/backend/interfaces/api/routers/settings.py` — settings schema + values endpoints.
- `apps/backend/interfaces/api/routers/ui.py` — tabs/workflows/blocks/presets persistence endpoints.
- `apps/backend/interfaces/api/routers/models.py` — model inventory + samplers/schedulers/embeddings + engine capabilities endpoints.
- `apps/backend/interfaces/api/routers/paths.py` — `apps/paths.json` endpoints.
- `apps/backend/interfaces/api/routers/options.py` — options store read/update/validate endpoints.
- `apps/backend/interfaces/api/routers/tasks.py` — task status/SSE/output endpoints.
- `apps/backend/interfaces/api/routers/tools.py` — GGUF converter + file browser endpoints.
- `apps/backend/interfaces/api/routers/generation.py` — txt2img/img2img/txt2vid/img2vid/vid2vid endpoints.
- `apps/backend/interfaces/api/file_metadata.py` — GGUF/SafeTensors header readers for `/api/models/file-metadata` (UI/debug).
- `apps/backend/interfaces/api/path_utils.py` — repo-relative path normalization helpers.
- `apps/backend/interfaces/api/json_store.py` — JSON load/save helpers for persistence files.
- `apps/backend/interfaces/api/task_registry.py` — in-process task registry (SSE queue + cancel flags).
- `apps/backend/interfaces/api/serializers.py` — checkpoint serialization helper.

## Notes
- `run_api.py` is composition-only: it wires routers and mounts the UI; route logic lives in `routers/`.
- Task state is centralized in `task_registry.py` so generation + tasks routers share cancellation/status logic.
- `/api/models/file-metadata` is intended for UI/debug; it returns `flat` plus a nested view of dotted keys. Codex-generated GGUF files use `model.*`, `codex.*`, and `gguf.*` keys (no legacy `general.*` provenance fields).
- 2026-01-18: `/api/models` checkpoint serialization now includes `core_only`, `core_only_reason`, and optional `family_hint` so the UI can stop guessing core-only status from filename suffixes alone.
- 2026-01-18: `/api/engines/capabilities` now also includes `engine_id_to_semantic_engine` so UI callers can keep engine-id and semantic-engine key spaces explicit.
