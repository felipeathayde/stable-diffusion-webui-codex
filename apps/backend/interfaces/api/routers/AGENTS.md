# apps/backend/interfaces/api/routers Overview
<!-- tags: backend, api, fastapi, routers -->
Date: 2026-01-08
Owner: Backend API Maintainers
Last Review: 2026-01-24
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
- 2026-01-18: `models.py` now includes backend `asset_contracts` in `/api/engines/capabilities` so the UI can gate required VAE/text encoder selection from a single contract source.
- 2026-01-18: `generation.py` enforces image asset requirements via `apps/backend/core/contracts/asset_requirements.py` and keeps engine registration lazy (avoids torch-heavy startup for non-generation endpoints).
- 2026-01-18: `generation.py` `vid2vid.method="wan_animate"` enforces repo-scoped paths under `CODEX_ROOT` (requires `vid2vid_model_dir`; stage `model_dir` must exist under the repo root).
- 2026-01-21: WAN stage LoRA selection is sha-only via `lora_sha` (sha → `.safetensors`); stage `lora_path` is rejected.
- 2026-01-21: Video tasks now honor Smart flags (`smart_offload`/`smart_fallback`/`smart_cache`) by propagating them into requests and applying `smart_runtime_overrides(...)` inside the video worker thread.
- 2026-01-23: `generation.py` enforces WAN video `height/width % 16 == 0` (txt2vid/img2vid/vid2vid; Diffusers parity) to avoid silent patch-grid cropping in the WAN22 runtime.
- 2026-01-23: WAN `%16` validation errors include an explicit `WIDTHxHEIGHT` + suggested rounded-up dimensions (for direct API callers; the UI snaps dims before POST).
- 2026-01-24: `settings.py` serves schema from the generated registry (JSON fallback) with no legacy static fallback; `run_api.py` prunes `apps/settings_values.json` against the registry on startup.
- 2026-01-24: `options.py` applies `codex_attention_backend` immediately via `memory_management.set_attention_backend(...)` (unloads/reinitializes runtime memory config; no backend restart required).
- 2026-01-24: `generation.py` applies live preview interval/method via thread-local `LivePreviewTaskConfig.runtime_overrides()` (no process-global `os.environ` mutation; avoids cross-request drift).
