# apps/backend/interfaces/api/routers Overview
<!-- tags: backend, api, fastapi, routers -->
Date: 2026-01-08
Last Review: 2026-02-18
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
- `apps/backend/interfaces/api/routers/supir.py` — SUPIR enhance endpoints (tasks + model diagnostics).
- `apps/backend/interfaces/api/routers/upscale.py` — upscalers inventory + remote downloads + standalone upscaling endpoints.
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
- 2026-01-21: Video tasks honor Smart flags via persisted options (`codex_smart_offload`/`codex_smart_fallback`/`codex_smart_cache`), propagating effective values into requests and applying `smart_runtime_overrides(...)` inside the video worker thread.
- 2026-01-23: `generation.py` enforces WAN video `height/width % 16 == 0` (txt2vid/img2vid/vid2vid; Diffusers parity) to avoid silent patch-grid cropping in the WAN22 runtime.
- 2026-01-23: WAN `%16` validation errors include an explicit `WIDTHxHEIGHT` + suggested rounded-up dimensions (for direct API callers; the UI snaps dims before POST).
- 2026-01-24: `settings.py` serves schema from the generated registry (JSON fallback) with no legacy static fallback; `run_api.py` prunes `apps/settings_values.json` against the registry on startup.
- 2026-01-24: `options.py` applies `codex_attention_backend` immediately via `memory_management.set_attention_backend(...)` (unloads/reinitializes runtime memory config; no backend restart required).
- 2026-01-24: `generation.py` applies live preview interval/method via thread-local `LivePreviewTaskConfig.runtime_overrides()` (no process-global `os.environ` mutation; avoids cross-request drift).
- 2026-01-25: `generation.py` now accepts `clip_skip` in `[0..12]` for txt2img (0 = “use default”), and validates `img2img_clip_skip` in `[0..12]` when provided.
- 2026-01-27: Video endpoints accept optional `video_return_frames` and pass it through request extras; WAN txt2vid/img2vid/vid2vid results can include `video {rel_path,mime}` for the UI player and omit frames by default.
- 2026-01-28: `generation.py` accepts `extras.zimage_variant="turbo"|"base"` (and `img2img_extras.zimage_variant`) and forwards it into `engine_options["zimage_variant"]` so the orchestrator can reload Z-Image when the variant changes.
- 2026-01-29: `generation.py` `img2img_*` now accepts explicit mask/inpaint controls when `img2img_mask` is provided (`img2img_mask_enforcement`, `img2img_inpaint_full_res[_padding]`, `img2img_inpainting_fill`, `img2img_inpainting_mask_invert`, `img2img_mask_blur[_x/_y]`, `img2img_mask_round`).
- 2026-01-29: `tools.py` adds `POST /api/tools/pnginfo/analyze` to extract PNG text metadata for the `/pnginfo` UI (no file persistence).
- 2026-01-29: `tools.py` CodexPack v1 output can be produced either as the primary output of `POST /api/tools/convert-gguf` (`codexpack_v1=true`; `output_path=*.codexpack.gguf`; base GGUF is temp-only and deleted on success) or from an existing base GGUF via `POST /api/tools/codexpack/pack-v1` (Z-Image Base/Turbo; `Q4_K`; Comfy Layout metadata required).
- 2026-01-30: `tools.py` GGUF conversion jobs now set `job["error"]` before flipping `job["status"]="error"` to avoid clients observing an error state with a missing error message.
- 2026-01-31: `generation.py` now delegates the txt2img/img2img task worker boilerplate (status/progress/result/end, engine options build, PNG encoding) to `apps/backend/interfaces/api/tasks/generation_tasks.py` to reduce drift and keep routers thin.
- 2026-02-03: Generation request contract uses hires naming only: txt2img uses `extras.hires` and img2img uses `img2img_hires_*` for the second pass.
- 2026-02-03: `upscale.py` validates `upscalers/manifest.json` (schema v1) and enriches `/api/upscalers/remote` results with two categories (curated vs other files) via `weights[].curated` + `weights[].meta`, while preserving raw listing fallback.
- 2026-02-05: `models.py` engine capabilities now include `anima` in `engine_id_to_semantic_engine` mapping (semantic tag `anima`).
- 2026-02-06: `models.py` engine capabilities now include backend-owned `dependency_checks` (ready + row list) so frontend tabs can render readiness panels and disable generation when dependencies are missing.
- 2026-02-06: `models.py` `engine_id_to_semantic_engine` now also maps `flux1_fill -> flux1` to keep strict frontend known-id semantic resolution in sync.
- 2026-02-05: `ui.py` tab allowlist now accepts `anima`; unknown/empty tab types are fail-loud (`400` for create payloads, `500` on invalid persisted `tabs.json` entries) instead of silent coercion to `sd15`.
- 2026-02-07: `generation.py` now validates Anima sampler selections early (txt2img/img2img + hires sampler overrides) against the semantic capability allowlist, returning HTTP 400 instead of late runtime `NotImplementedError`.
- 2026-02-08: `generation.py` now parses `extras.er_sde` / `img2img_extras.er_sde` strictly (unknown-key rejection, solver normalization, numeric bounds/finite checks), enforces ER-SDE Anima-only release scope on base+hires sampler fields, and validates prompt `<sampler:...>` control tags against release scope/allowlists.
- 2026-02-08: `generation.py` hires parser hardening now keeps allowlist and consumed keys aligned for `extras.hires.distilled_cfg`, validates hires numeric fields (`denoise`, `scale`, `cfg`, `distilled_cfg`) via `_require_float_field`, and rejects non-finite values (`NaN`, `Infinity`) with fail-loud 400s.
- 2026-02-08: `generation.py` swap-model contract now uses `switch_at_step` (global `extras.refiner` + nested `extras.hires.refiner`) with strict pointer validation (`1 <= switch_at_step < total_steps`) so SDXL model swap runs on a step pointer, not a refiner step-count.
- 2026-02-08: `generation.py` img2img numeric parsing now uses `_require_float_field` for core + hires numeric fields (`img2img_cfg_scale`, `img2img_denoising_strength`, `img2img_hires_*` float controls), enforcing finite-number 400s for `NaN`/`Infinity`.
- 2026-02-09: `tasks.py` now parses `/api/tasks/{task_id}/cancel` mode through typed `TaskCancelMode` parsing (invalid values return HTTP 400), and emits gap/result/error/end payload type literals via `TaskEventType` enum values (wire format preserved). `generation.py` worker cancel checks now compare against `TaskCancelMode.IMMEDIATE`.
- 2026-02-10: `generation.py` now uses internal parser DTO seams for core txt2img/img2img normalization (`_Txt2ImgPayloadDTO`, `_Img2ImgCoreDTO` + parser helpers), keeping extras/hires/asset-contract flow unchanged while reducing dict-heavy parser blocks.
- 2026-02-10: `generation.py` now extends parser DTO seams into active video modes (`_parse_txt2vid_core_dto`, `_parse_img2vid_core_dto`) while preserving WAN sha-only asset contracts and current request/task/SSE behavior (validated by video request-capture parity tests).
- 2026-02-10: `tools.py` now enforces typed job lifecycle internals for GGUF/CodexPack jobs (`_ToolJobStatus`, `_ToolJobState`, typed controls + guarded transitions), while preserving status endpoint wire payload shape.
- 2026-02-11: `generation.py` now resolves `extras.vae_sha` through a VAE-only inventory helper (`resolve_vae_path_by_sha`) and returns HTTP 409 when a SHA maps to a non-VAE asset path, preventing Flux core-only VAE misselection from leaking into loader missing-key noise.
- 2026-02-11: WAN video routes (`txt2vid`/`img2vid`) now resolve `wan_vae_sha` through VAE-only ownership (`resolve_vae_path_by_sha`) and normalize to a validated VAE bundle directory (`config.json` + weights file required), rejecting non-VAE SHA resolution and invalid bundles with HTTP 409 before runtime fallback loops.
- 2026-02-11: WAN video routes (`txt2vid`/`img2vid`) now also accept file-based WAN VAE SHA resolution when config is available via sibling `config.json` or vendored metadata (`wan_metadata_dir/vae/config.json`), while keeping fail-loud 409 for missing config sources, missing bundle weights, and non-VAE SHA ownership violations.
- 2026-02-15: `generation.py` enforces the strict generation settings contract: top-level `smart_*` payload keys are rejected and `settings_revision` must match persisted `codex_options_revision` (HTTP 409 includes `current_revision` + `provided_revision`).
- 2026-02-15: `options.py` `POST /api/options` now returns apply metadata arrays (`applied_now[]`, `restart_required[]`) with per-key reason strings; runtime-memory keys are classified as hot-applied.
- 2026-02-15: `generation.py` video task worker now emits contract-trace JSONL events (opt-in via `CODEX_TRACE_CONTRACT`) with stage/action/device + prompt hash only (no raw prompt fields).
- 2026-02-15: `tasks.py` now sanitizes terminal `entry.error` before returning `/api/tasks/{task_id}` and SSE `error` events, preventing raw exception text leaks to clients.
- 2026-02-15: `generation.py`/`upscale.py`/`supir.py` now sanitize synchronous `HTTPException.detail` paths (and `upscale.py` manifest parse error fields) through `public_http_error_detail(...)`, so API callers do not receive raw exception strings.
- 2026-02-16: `generation.py` video worker now logs typed `EngineExecutionError` explicitly to API console logs before writing sanitized task error payloads (keeps local debugging signal while preserving public error contract).
- 2026-02-16: `generation.py` video worker immediate-cancel path now drains orchestrator iterators instead of returning early, so teardown/finalizers complete before `release_inference_gate()`; video smart-flag parsing now reuses the shared strict helper from `tasks/generation_tasks.py`.
- 2026-02-16: WAN22 video request allowlists are now owned by model keymap module `apps/backend/runtime/state_dict/keymap_wan22_transformer.py` (`WAN22_REQUEST_KEYS`), not by payload type definitions.
- 2026-02-16: `tasks.py` cancellation endpoint now rejects `mode="after_current"` with HTTP 400 until worker-level deferred-cancel semantics are implemented (fail-loud contract).
- 2026-02-16: `models.py` prompt-token endpoint now recognizes `wan22_animate_14b`; `ui.py` tab-type normalization accepts `wan22_animate_14b` and normalizes all WAN aliases to `wan`.
- 2026-02-17: `generation.py` WAN variant resolution now preserves 14B identity across repo/path hints (expanded token set + 14B-first heuristics) and avoids silent 14B→5B collapse during engine-key resolution.
- 2026-02-17: `generation.py` keeps animate metadata hints remapped to the task-capable `wan22_14b` lane for txt2vid/img2vid while preserving the requested variant metadata (`wan_engine_variant`).
- 2026-02-17: `generation.py` WAN video core validation now enforces frame domain `4n+1` in `[9,401]`, accepts strict `gguf_attention_mode` (`global|sliding`), and validates/forwards img2vid chunk controls (`img2vid_chunk_frames`, `img2vid_overlap_frames`, `img2vid_anchor_alpha`, `img2vid_chunk_seed_mode`) fail-loud.
- 2026-02-17: `generation.py` video worker now resolves core dtype overrides from persisted options (`codex_core_compute_dtype` with fallback to `codex_core_dtype`) and forwards `engine_options["dtype"]` to WAN loads; invalid option types/values fail loud.
- 2026-02-17: `generation.py` resolves options/dtype inside the existing prepare-guard block so invalid options still follow terminal task cleanup (`entry.error`, `mark_finished(false)`, `unregister_task`) with no leaked task registry entries.
- 2026-02-18: `generation.py` now resolves `extras.lora_sha`/`img2img_extras.lora_sha` into `lora_path` overrides using inventory ownership from `inventory.loras` (must resolve to known `.safetensors` LoRA paths; non-LoRA SHA resolution fails with HTTP 409).
- 2026-02-18: `generation.py` now parses optional `extras.guidance` / `img2img_extras.guidance` with strict key/range validation (`apg_*`, `guidance_rescale`, `cfg_trunc_ratio`, `renorm_cfg`) and forwards the parsed object through override settings for sampler policy resolution.
