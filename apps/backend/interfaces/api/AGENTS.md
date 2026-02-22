# apps/backend/interfaces/api Overview
<!-- tags: backend, api, fastapi, routers -->
Date: 2026-01-08
Last Review: 2026-02-21
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
- `apps/backend/interfaces/api/public_errors.py` — public-safe error formatters for task channels and synchronous HTTP details.
- `apps/backend/interfaces/api/tasks/generation_tasks.py` — shared generation task worker helpers (image task runners + engine options + PNG encoding).
- `apps/backend/interfaces/api/serializers.py` — checkpoint serialization helper.
- `apps/backend/interfaces/api/upscalers_manifest.py` — `upscalers/manifest.json` schema validation/normalization (used by `/api/upscalers/remote`).
- `apps/backend/interfaces/api/dependency_checks.py` — backend-owned dependency-check builder used by `/api/engines/capabilities`.

## Notes
- `run_api.py` is composition-only: it wires routers and mounts the UI; route logic lives in `routers/`.
- Task state is centralized in `task_registry.py` so generation + tasks routers share cancellation/status logic.
- `/api/models/file-metadata` is intended for UI/debug; it returns `flat` plus a nested view of dotted keys. Codex-generated GGUF files use `model.*`, `codex.*`, and `gguf.*` keys (no legacy `general.*` provenance fields).
- 2026-01-18: `/api/models` checkpoint serialization now includes `core_only`, `core_only_reason`, and optional `family_hint` so the UI can stop guessing core-only status from filename suffixes alone.
- 2026-01-18: `/api/engines/capabilities` now also includes `engine_id_to_semantic_engine` so UI callers can keep engine-id and semantic-engine key spaces explicit.
- 2026-01-20: Removed unreferenced API helper modules (`media_helpers.py`, `script_models.py`) (no call sites).
- 2026-01-21: WAN stage LoRA inputs are sha-only (`lora_sha`); raw-path stage `lora_path` is rejected during payload normalization/validation.
- 2026-01-24: Settings schema/values are now strict: schema is served from the generated registry (JSON fallback), and persisted values are pruned against the registry on startup (unknown keys dropped; invalid values clamped).
- 2026-01-25: `run_api.py` migrated the deprecated `@app.on_event("startup")` hook to FastAPI lifespan handlers (removes DeprecationWarning).
- 2026-01-31: Added `interfaces/api/tasks/` to keep routers thin by centralizing shared generation task worker boilerplate (status/progress/result/end + engine options build for image modes).
- 2026-02-06: Added backend-owned `dependency_checks` contract for `/api/engines/capabilities` (ready + per-row checks), built in `dependency_checks.py`.
- 2026-02-06: `/api/engines/capabilities` key-space map now includes `flux1_fill -> flux1` in `engine_id_to_semantic_engine` for strict frontend taxonomy mapping parity.
- 2026-02-08: Swap-model request contract now uses `switch_at_step` in both `extras.refiner` and `extras.hires.refiner` (step-pointer semantics, not step-count semantics) with strict bounds validation in `routers/generation.py`.
- 2026-02-08: Img2img numeric parsing now uses finite-float validation for core + hires float fields in `routers/generation.py` (rejects `NaN`/`Infinity` with HTTP 400).
- 2026-02-09: Task contracts are now typed in `task_registry.py` (`TaskEventType`, `TaskCancelMode`, `TaskStatusStage`) with strict non-terminal event normalization and fail-loud cancel-mode parsing (`immediate`/`after_current` only).
- 2026-02-10: `/api/engines/capabilities` now emits `asset_contracts` keyed by semantic engine (owner-resolved from canonical engine ids) so frontend semantic gating cannot drift from alias-heavy `engine_id_to_semantic_engine` maps.
- 2026-02-10: `dependency_checks.py` now resolves semantic asset contracts only via `contract_owner_for_semantic_engine(...)` (no local semantic-map duplication), so readiness rows fail loud if semantic/contract ownership drifts.
- 2026-02-15: Generation payloads now require `settings_revision` to match persisted options revision; stale requests fail with HTTP 409 (`current_revision` + `provided_revision`) and top-level `smart_*` payload keys are rejected.
- 2026-02-15: `run_api.py` startup settings normalization preserves `codex_options_revision` while pruning unknown/invalid persisted keys.
- 2026-02-15: `POST /api/options` responses now include `applied_now[]` and `restart_required[]` arrays with per-key reason metadata.
- 2026-02-15: `run_api.py` now publishes launcher trace toggles (`trace_contract`, `trace_profiler`) through bootstrap env keys (`CODEX_TRACE_CONTRACT`, `CODEX_TRACE_PROFILER`) and maps profiler toggle to `CODEX_PROFILE=1` for runtime diagnostics.
- 2026-02-15: Task error payloads now pass through `public_errors.py`; API task status/SSE channels expose public-safe terminal messages (`cancelled`/`out of memory`/stable error id) instead of raw exception text.
- 2026-02-15: `public_errors.py` also sanitizes synchronous HTTP error details for generation/upscale/supir routes (`public_http_error_detail`), removing raw exception text from `HTTPException.detail` and `/api/upscalers/remote` manifest parse errors while preserving actionable OOM classification.
- 2026-02-15: `public_errors.py` now keeps `EngineExecutionError` messages visible in task channels using stable `engine error: ...` formatting (idempotent on replay/snapshot re-serialization), so frontend task error panels can surface actionable runtime failures instead of opaque `internal error (error_id=...)`.
- 2026-02-16: Generation task workers now also emit explicit API-console logs for typed `EngineExecutionError` (`task_id` + `mode` + `engine` + message) before public-error sanitization, so local runtime failures remain visible in backend logs without changing task/SSE payload contracts.
- 2026-02-18: `run_api.py` phase-1 logging cleanup removed the `logging.basicConfig(...)` fallback in `ensure_initialized()` (fail-loud on logging bootstrap failure) and migrated startup/settings/port-guard/init console `print(...)` messages to structured logger calls.
- 2026-02-18: `run_api.py` bootstrap env publication now also exports non-default LoRA loader toggles (`CODEX_LORA_MERGE_MODE`, `CODEX_LORA_REFRESH_SIGNATURE`) alongside `CODEX_LORA_APPLY_MODE`, preserving CLI/env/settings precedence without mutating process `os.environ`.
- 2026-02-20: `json_store.py` is now fail-loud for persistence faults: `_load_json` returns `{}` only for missing files and raises on parse/read/type violations; `_save_json` raises on write/serialization failures (no best-effort swallow).
- 2026-02-21: `run_api.py` startup settings normalization now parses checkbox values via shared strict bool parser and fails startup on invalid checkbox literals (no silent coercion of unknown strings to `False`).
- 2026-02-21: UI persistence routes now fail loud on malformed `tabs.json`/`workflows.json`/`presets.json` payloads (no silent default/empty remap), and `/api/options` now rejects out-of-range numeric values instead of silently clamping (aligned with `/api/options/validate`).
- 2026-02-21: `run_api.py` checkbox startup normalization now canonicalizes persisted checkbox values to strict `bool` type (including `0/1` -> `False/True`) to prevent numeric-bool type drift in `settings_values.json`.
- 2026-02-22: `routers/system.py` adds `POST /api/obliterate-vram` (quick-settings VRAM cleanup) with safe default behavior: internal runtime cleanup always runs, external process termination is opt-in via `external_kill_mode='all'`, and critical/process-self protections are enforced with structured report output for UI feedback.
