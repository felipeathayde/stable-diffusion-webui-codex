<!-- tags: webui, architecture, reference, map -->
# WebUI Subsystem Map (single-file reference)
Date: 2026-01-02
Last Review: 2026-01-18
Version: 2026-01-18
Status: Draft

This file is the **top-level map** of how the WebUI is assembled and how requests flow through the system.
For deeper specs, follow the links under “Canonical reference docs”.

## Index (search-first)
- `Read-first contracts` (no guessing): see “Read-first contracts” below.
- `Anti-drift` (common wrong assumptions): search for `DRIFT-`.
- `API routes` (source of truth): search for `API ROUTES (routers/*.py)`.
- `Model parts matrix` (what is required): search for `MODEL PARTS MATRIX`.
- `Why lists are empty` (common UI symptom): search for `Why lists can look “empty”`.
- `Debugging checklist`: search for `Debugging checklist`.

## Canonical reference docs (source of truth for contracts)
- Model assets (discovery → inventory → SHA selection): `.sangoi/reference/models/model-assets-selection-and-inventory.md`
- API tasks + SSE streaming: `.sangoi/reference/api/tasks-and-streaming.md`
- UI model tabs + QuickSettings + generation flow: `.sangoi/reference/ui/model-tabs-and-quicksettings.md`

## Read-first contracts (do not infer)
- **No free-form model paths in normal image generation.** UI sends stable IDs only:
  - Checkpoints: short hash (`10` hex) in `model` (or `extras.model_sha`)
  - External assets: full sha256 (`64` hex) in `extras.*_sha`
- **GGUF engines need complements.** Flux/ZImage/WAN GGUF checkpoints are core-only, so TE/VAE (and WAN stage models) must be provided explicitly.
- **“Built-in” means “no override”.** Valid for monolithic checkpoints (SD15/SDXL). Engines that require external assets must error instead of guessing.
- **Device is required.** Every generation request must include an explicit device (no options snapshot fallback).
- **Task streams always end.** SSE `/api/tasks/{id}/events` must terminate with an `end` event.

## Anti-drift (common critical wrong assumptions from session reasoning)
These are the recurring “dangerous assumptions” that showed up in session reasoning.
They are not theoretical — if you follow the drift, you end up debugging the wrong thing or (worse) implementing silent fallbacks that this WebUI explicitly rejects.

### DRIFT-API-PREFIX: Assuming legacy endpoints exist (`/sdapi/v1/*`, `/codex/api/v1/*`, `/api/v1/*`)
- Wrong assumption: the WebUI uses a legacy WebUI-compatible prefix (or a “codex_api” router) for generation.
- Correct contract: current UI/backends use `/api/*` only; don’t invent endpoints — confirm in `apps/backend/interfaces/api/routers/*`.
- Quick verify: `rg -n \"@router\\.(get|post|patch|delete)\\(\\\"/api\" apps/backend/interfaces/api/routers -S`.

### DRIFT-ASSETS-OPTIONAL: Treating VAE / text encoders as optional for core-only engines
- Wrong assumption: “ZImage can proceed without a VAE”, or “Flux text encoders are optional”, or “return None when missing and keep going”.
- Correct contract (backend): requirements are contract-driven and enforced at the API boundary (no inference/fallbacks):
  - Owner: `apps/backend/core/contracts/asset_requirements.py` (`contract_for_request(engine_id, checkpoint_core_only)`).
  - Enforcer: `apps/backend/interfaces/api/routers/generation.py` (builds 400/409 errors from the same contract).
  For `engine_id in ("flux1", "flux1_kontext", "zimage")`, **both** VAE and TE must be provided via SHA:
  - txt2img: `extras.vae_sha` + `extras.tenc_sha`
  - img2img: `img2img_extras.vae_sha` + `img2img_extras.tenc_sha`
  Missing → `400`, unknown SHA → `409`. Source: `apps/backend/interfaces/api/routers/generation.py`.
- Correct contract (UI): if the engine requires an asset and the UI cannot resolve a SHA, it must block submission (no guessing).

### DRIFT-INFER-REQUIREMENTS: Inferring requirements from pipeline config / latent format
- Wrong assumption: “If pipeline_config has a `vae` key, then VAE must be optional/required”, or “infer VAE need from latent format and guess exceptions.”
- Correct contract: requirements are enforced at the API boundary by the shared asset contract (engine-id + core-only), not inferred from pipeline configs.
  Source: `apps/backend/core/contracts/asset_requirements.py` + `apps/backend/interfaces/api/routers/generation.py`.

### DRIFT-FLUX-TE-COUNT: Treating “text encoder” as a single value for Flux/Kontext
- Wrong assumption: “pick the first text encoder” / treat TE as a single string.
- Correct contract: Flux/Kontext require **exactly 2** encoders (`clip_l` + `t5xxl`) via `extras.tenc_sha` (array of 2 sha256 strings).
  The API maps order-independently using the header-only slot classifier in `apps/backend/core/contracts/text_encoder_slots.py` (no reliance on list order). Source: `apps/backend/interfaces/api/routers/generation.py`.

### DRIFT-SENTINELS: Treating `Automatic` / `Built-in` as a valid fallback for required assets
- Wrong assumption: “if user leaves Built-in/Automatic, just don’t send it and it’ll work.”
- Correct contract:
  - For SD15/SDXL, `Built-in` means “do not override” (checkpoint-embedded assets).
  - For Flux/ZImage/WAN GGUF, there is no “built-in” TE/VAE inside the core checkpoint — missing required SHAs must fail (UI should guide the user to select a real asset).

### DRIFT-NO-FALLBACK: Implementing silent fallbacks for missing/unknown model parts
- Wrong assumption: “pick something” when missing (first VAE, first TE, default checkpoint, etc.).
- Correct contract: **no silent fallbacks** for required model parts.
  - Unknown SHA must error (HTTP `409`).
  - Missing required SHA must error (HTTP `400`).

### DRIFT-RAW-PATHS: Treating internal `*_path` fields as client inputs
- Wrong assumption: the client can send `extras.vae_path` / `extras.tenc_path` (or `img2img_extras.vae_path` / `img2img_extras.tenc_path`) as free-form filesystem paths.
- Correct contract: client requests must use SHA fields only (`*_sha`). The backend resolves SHA → path and only then injects `*_path` into engine options server-side.
  - txt2img: `extras` schema does not accept raw `*_path` keys.
  - img2img: backend explicitly rejects raw `*_path` keys in `img2img_extras`.

### DRIFT-WAN-PATHS: Assuming WAN video uses raw `wan_*_path` inputs
- Wrong assumption: WAN complements are sent as paths (e.g. `wan_vae_path`, `wan_text_encoder_path`) and stage selection uses `wan_high.model_dir` / `wan_low.model_dir`.
- Correct contract (current backend): WAN video endpoints resolve model parts by SHA only:
  - Stage models: `wan_high.model_sha` / `wan_low.model_sha` (must resolve to `.gguf`)
  - Complements: `wan_vae_sha` + `wan_tenc_sha` (VAE + `.safetensors`)
  - Metadata/tokenizer dirs remain explicit paths (`wan_metadata_dir` or `wan_tokenizer_dir`), resolved relative to `CODEX_ROOT` when repo-relative.

## API ROUTES (routers/*.py)
Source of truth: `apps/backend/interfaces/api/routers/*` (router decorators).

Generation tasks (async; return `{task_id}`):
- `POST /api/txt2img`
- `POST /api/img2img`
- `POST /api/txt2vid`
- `POST /api/img2vid`
- `POST /api/vid2vid`

Task lifecycle + streaming (SSE):
- `GET /api/tasks/{task_id}`
- `GET /api/tasks/{task_id}/events`
- `POST /api/tasks/{task_id}/cancel`

Models + assets:
- `GET /api/models` (`?refresh=1`)
- `GET /api/models/file-metadata`
- `GET /api/models/checkpoint-metadata`
- `GET /api/models/inventory` (`?refresh=true`)
- `POST /api/models/inventory/refresh`
- `GET /api/paths`
- `POST /api/paths`

Options + settings:
- `GET /api/options`
- `GET /api/options/keys`
- `GET /api/options/snapshot`
- `GET /api/options/defaults`
- `POST /api/options`
- `POST /api/options/validate`
- `GET /api/settings/schema`
- `GET /api/settings/values`

UI persistence:
- `GET /api/ui/tabs`
- `POST /api/ui/tabs`
- `PATCH /api/ui/tabs/{tab_id}`
- `POST /api/ui/tabs/reorder`
- `DELETE /api/ui/tabs/{tab_id}`
- `GET /api/ui/workflows`
- `POST /api/ui/workflows`
- `PATCH /api/ui/workflows/{wf_id}`
- `DELETE /api/ui/workflows/{wf_id}`
- `GET /api/ui/blocks`
- `GET /api/ui/presets`
- `POST /api/ui/presets/apply`

Catalogs:
- `GET /api/samplers`
- `GET /api/schedulers`
- `GET /api/vaes`
- `GET /api/text-encoders`
- `GET /api/embeddings`
- `GET /api/loras`
- `GET /api/loras/selections`
- `POST /api/loras/apply`

Tools + output:
- `POST /api/models/load`
- `POST /api/models/unload`
- `GET /api/engines/capabilities`
- `POST /api/tools/convert-gguf`
- `GET /api/tools/convert-gguf/{job_id}`
- `GET /api/tools/browse-files`
- `GET /api/output/{rel_path:path}`
- `GET /api/health`
- `GET /api/version`
- `GET /api/memory`

## MODEL PARTS MATRIX (what must be explicit)
This is the “don’t drift” table for model-part requirements as enforced by the API layer.
Source of truth:
- Per-engine requirements: `apps/backend/core/contracts/asset_requirements.py` (also exposed to the UI via `GET /api/engines/capabilities` → `asset_contracts`).
- Core-only hint: `GET /api/models` → `models[].core_only` (currently `.gguf` ⇒ core-only).

- SD15 / SD20 / SDXL / SDXL-refiner / SD3.5 (`engine_id: sd15|sd20|sdxl|sdxl_refiner|sd35`)
  - Checkpoint: monolithic `.safetensors` (includes UNet/TE/VAE).
  - VAE/TE: optional overrides only; `Built-in` means “no override”.
  - Core-only checkpoints (`models[].core_only=true`) require external assets per contract (VAE + TE count/kind).
- Flux / Kontext (`engine_id: flux1|flux1_kontext`)
  - External-assets-first: always requires complements.
  - Requires: `extras.vae_sha` + `extras.tenc_sha` (array of 2; CLIP + T5).
- ZImage (`engine_id: zimage`)
  - External-assets-first: always requires complements.
  - Requires: `extras.vae_sha` + `extras.tenc_sha` (single; Qwen).
- Chroma (`engine_id: flux1_chroma`)
  - Safetensors treated as monolithic; core-only checkpoints require external VAE + 1 T5.
- WAN (video endpoints)
  - GGUF sha-only (`method != "wan_animate"`):
    - Stage models: `wan_high.model_sha` + `wan_low.model_sha` (sha → `.gguf`)
    - Complements: `wan_vae_sha` + `wan_tenc_sha` (sha → VAE + `.safetensors|.gguf`)
    - Metadata: `wan_metadata_repo` (preferred) or `wan_metadata_dir` (path)
  - `wan_animate` (Diffusers dir; path-based but repo-scoped):
    - Requires `vid2vid_model_dir` (directory under `CODEX_ROOT`)
    - Stage overrides (`wan_high/wan_low`) accept `model_dir` (file/dir) and optional `lora_path` (file), all under `CODEX_ROOT`
    - Optional `wan_metadata_dir` / `wan_tokenizer_dir` must be directories under `CODEX_ROOT`

## Repo anchors (constants and config)
- **`CODEX_ROOT`** is the repository root resolved by the backend (and the launcher). It is used to:
  - Normalize API paths to be repo-relative when safe.
  - Define safe roots for output serving and uploads.
  - Anchor defaults (e.g., curated model roots under `models/**`).
  - Code: `apps/backend/interfaces/api/run_api.py` (composition; passes `CODEX_ROOT` into routers)
- Model roots configuration: `apps/paths.json`
  - Served/updated via `GET/POST /api/paths`.
  - Used by backend scanners to find checkpoints and assets.
  - Also used by the UI to constrain dropdowns; if `/api/paths` is empty or wrong, some per-family selectors will render empty even when inventory has items.
- Settings persistence:
  - Options values file: `apps/settings_values.json` (backend-managed).
  - Settings schema: `apps/backend/interfaces/schemas/settings_schema.json` (preferred) with legacy fallback.

## End-to-end flow (txt2img/img2img)
1) UI loads lists/options (models, inventory, samplers, schedulers, options).
2) User selects tab params + QuickSettings overrides (model/device/assets/etc).
3) UI builds payload and submits `POST /api/txt2img` (or `/api/img2img`).
4) Backend returns `{task_id}` immediately, then runs inference in a worker thread.
5) UI subscribes to SSE `/api/tasks/{task_id}/events` and renders:
   - `status` → queued/running
   - `progress` (+ optional live preview fields)
   - `result` (images + info)
   - `error`
   - `end` (final; closes stream)

## Backend (API + orchestration)
### API entrypoint
- FastAPI composition: `apps/backend/interfaces/api/run_api.py`
- Route logic + payload enforcement: `apps/backend/interfaces/api/routers/*` (generation: `apps/backend/interfaces/api/routers/generation.py`)

Key API families:
- **Generation tasks**: `POST /api/txt2img`, `POST /api/img2img`, `POST /api/txt2vid`, `POST /api/img2vid`, `POST /api/vid2vid`
- **Task control**: `GET /api/tasks/{id}`, `GET /api/tasks/{id}/events` (SSE), `POST /api/tasks/{id}/cancel`
- **Model listing**: `GET /api/models` (`?refresh=1`)
- **Metadata (UI/debug)**: `GET /api/models/file-metadata`, `GET /api/models/checkpoint-metadata`
- **Inventory listing**: `GET /api/models/inventory`, `POST /api/models/inventory/refresh`
- **Options / settings**: `GET/POST /api/options`, `GET /api/options/snapshot`, `GET /api/settings/schema`, `GET /api/settings/values`
- **UI persistence**: `GET/POST/PATCH/DELETE /api/ui/tabs`, `GET/POST /api/ui/workflows`, `GET /api/ui/blocks`, `GET/POST /api/ui/presets`
- **Tools**: `POST /api/tools/convert-gguf`, `GET /api/tools/convert-gguf/{job_id}`, `GET /api/tools/browse-files`
- **Safe output**: `GET /api/output/{rel_path:path}` (scoped to `CODEX_ROOT/output`)

### Hard invariants enforced by the backend
- **Explicit device selection** is required in generation payloads (no options snapshot fallback).
  - Enforced by: `_require_explicit_device(...)` in `apps/backend/interfaces/api/routers/generation.py`
- **No silent fallbacks for required model parts** (VAEs / text encoders / GGUF stage models).
  - Missing required SHA fields → `400`
  - Unknown SHA → `409`

### Orchestration boundary
- Inference orchestrator: `apps/backend/core/orchestrator.py`
  - Routes `(task_type, engine_key, request, model_ref, engine_options)` to the correct runtime engine.
  - Coordinates load/unload and memory behavior across jobs:
    - Purges VRAM on model signature changes (checkpoint/text-encoder selection swaps).
    - On engine load/execution failures, scrubs tracebacks and performs an aggressive purge (unload all models + clear engine cache + GC + CUDA cache/IPCs) so the backend can recover without restart.
- Bundle-aware engine loader (used by use-cases and tooling): `apps/backend/core/engine_loader.py`
  - Resolves a `DiffusionModelBundle`, instantiates the matching engine, loads weights, and applies runtime options for diffusers-backed pipelines.

### Memory management (VRAM/RAM)
- Runtime memory manager facade: `apps/backend/runtime/memory/memory_management.py`
  - Exposes the active `CodexMemoryManager` as `memory_management.manager` and supports runtime device/dtype switching.
- Central memory policy + loaded-model registry: `apps/backend/runtime/memory/manager.py`
  - Owns the loaded-model registry, swap/offload policies, and best-effort cache eviction (`unload_all_models`, `soft_empty_cache`).

## Model discovery + inventory (hash/SHA-based selection)
This is the “weight selection layer” shared by UI and backend.

- Checkpoint registry scan (short hash for checkpoints):
  - Code: `apps/backend/runtime/models/registry.py`
  - Cache: `models/.hashes.json` (internal, generated; do not commit)
  - Notes: the registry stores a best-effort safetensors dtype hint (header-only) alongside hashes so loaders/UIs can default precision without loading full tensors.
- Inventory scan (full sha256 for external assets):
  - Code: `apps/backend/inventory/cache.py`
  - Resolution: SHA → path lookup used by API enforcement and engine options

Selection rules (high level):
- **Checkpoints**: UI typically sends a **10-char short hash** (or a label) via `model=...`.
- **External assets** (VAE/TE/LoRA/etc): UI sends **full sha256** via `extras.*_sha` fields.

### Why lists can look “empty” (QuickSettings filtering)
- Backend provides raw lists via `/api/models`, `/api/models/inventory`, and `/api/paths`.
- The UI applies per-family filtering using `/api/paths` roots (see `fileInPaths(...)` in `apps/interface/src/components/QuickSettingsBar.vue`).
- Common failure mode: inventory has files, but `/api/paths` returns empty roots → filtered Flux/ZImage TE/VAE lists become empty.

## Frontend (Vue UI)
### Router and surfaces
- Main router: `apps/interface/src/router.ts`
  - Model tabs: `/models/:tabId` (dynamic engine tabs)
  - Utilities: `/tools`, `/workflows`, `/xyz`, `/settings`, etc.

### Key state stores
- QuickSettings: `apps/interface/src/stores/quicksettings.ts`
  - Loads options/lists (`/api/models`, `/api/models/inventory`, `/api/options`, …)
  - Builds `sha256` lookup maps and resolves labels → sha
  - Owns cross-tab toggles (device, smart flags, streaming, etc.)
- Model tabs: `apps/interface/src/stores/model_tabs.ts`
  - Persists per-tab params (image tabs + WAN video tab)
  - Uses engine capabilities/defaults from `apps/interface/src/stores/engine_config.ts`

### API client and task streaming
- API wrapper: `apps/interface/src/api/client.ts`
  - Uses `EventSource` for `/api/tasks/{task_id}/events`
  - Closes stream on `{type:"end"}` and suppresses noisy onerror after normal close

### Payload builders (high-level)
- Image generation: `apps/interface/src/composables/useGeneration.ts`
  - Resolves inventory-backed selectors:
    - TE → `extras.tenc_sha` (string or list)
    - VAE → `extras.vae_sha` (required for some engines; optional override for SD/SDXL)
  - Blocks submission when an engine requires a field (e.g. requires VAE) and it cannot resolve a SHA.
- Video generation (WAN): `apps/interface/src/composables/useVideoGeneration.ts`
  - Submits video tasks + subscribes to the same task SSE stream shape.

## Tools subsystem (GGUF converter)
High-level contract: Tools are deterministic, auditable operations exposed via `/api/tools/*` (no silent fallbacks; fail loud).

End-to-end flow:
1) UI submits `POST /api/tools/convert-gguf` and polls `GET /api/tools/convert-gguf/{job_id}`.
2) Backend converts SafeTensors → GGUF (profile-driven planner + quantization policy), injects metadata, then verifies output.
3) Result artifacts are served via `/api/output/{rel_path}` (scoped to `CODEX_ROOT/output`).

Backend code map (source of truth):
- Tool entrypoint + orchestration: `apps/backend/runtime/tools/gguf_converter.py`
- Public types + progress: `apps/backend/runtime/tools/gguf_converter_types.py`
- Profile registry (layout/planner + per-model dtype policies): `apps/backend/runtime/tools/gguf_converter_profiles.py`
- Tensor planning + key remaps: `apps/backend/runtime/tools/gguf_converter_tensor_planner.py`
- Metadata injection helpers: `apps/backend/runtime/tools/gguf_converter_metadata.py`
- Vendored model metadata listing for UI presets: `apps/backend/runtime/tools/gguf_converter_model_metadata.py`

## Engine families (conceptual map)
The engine key and asset requirements are defined by backend capabilities + request payloads.

- **SD15 / SDXL** (monolithic safetensors by default)
  - UI may optionally include an external `vae_sha`, but “Built-in” means “use checkpoint-embedded assets”.
- **Flux / Kontext**
  - Flux img2img routes through the Kontext workflow engine (UI-level engine override).
  - TE overrides can be SHA-based and can represent multi-component encoders.
- **ZImage**
  - Uses inventory-backed selection for complementary assets when required by the engine spec.
- **WAN22**
  - Video endpoints share task streaming but have extra payload rules.
  - `vid2vid` is multipart upload and saves files under `CODEX_ROOT/tmp/uploads/vid2vid/` before running.
  - WAN (GGUF) selects model parts by SHA (no raw paths):
    - Stage models: `wan_high.model_sha` / `wan_low.model_sha` → must resolve to `.gguf`
    - Complements: `wan_vae_sha` + `wan_tenc_sha` → must resolve to VAE + `.safetensors`
    - Metadata/tokenizer dirs remain explicit paths (resolved relative to `CODEX_ROOT` when repo-relative).

### Engine assembly seams (spec + factory)
- Engine families standardize assembly via `spec.py` + `factory.py` modules under `apps/backend/engines/<family>/`:
  - `spec.py` defines typed runtime containers and `assemble_*` helpers.
  - `factory.py` is the family assembly boundary (builds runtime + `CodexObjects` from a `DiffusionModelBundle`).
- Factory plan: `.sangoi/plans/2026-01-03-engine-factory-standard-v1.md`.
- Runtime layout: model-family runtimes live under `apps/backend/runtime/families/<family>/` (plan/rationale: `.sangoi/plans/2026-01-17-backend-runtime-families-layout.md`).

## Security / path hygiene (what is allowed)
- API responses prefer repo-relative `path` values under `CODEX_ROOT` (avoid leaking host absolute paths).
- Output serving (`/api/output/{rel_path}`) is root-scoped and rejects traversal.
- Uploaded vid2vid files are saved under `CODEX_ROOT/tmp/uploads/vid2vid/` and best-effort cleaned up after the task finishes.

## Debugging checklist (common “why is it empty?” issues)
- `/api/paths` empty → QuickSettings filtering removes most Flux/ZImage assets; fix `apps/paths.json` and `CODEX_ROOT` resolution.
- Backend cannot see weights (WSL/mounts/sandbox) → `/api/models*` returns empty lists and `models/.hashes.json` may not exist.
- Force refresh: `/api/models?refresh=1` and `/api/models/inventory?refresh=true` (or `POST /api/models/inventory/refresh`).

## Where to extend next (if you want more coverage here)
- Workflows subsystem: UI + `/api/ui/workflows` integration and how workflows affect generation payloads.
- Tools subsystem: GGUF converter end-to-end (UI trigger → job polling → artifacts).
- Results/History subsystem: UI storage model and selection behavior across tabs.
