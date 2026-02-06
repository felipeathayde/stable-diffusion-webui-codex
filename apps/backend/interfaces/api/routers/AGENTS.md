# apps/backend/interfaces/api/routers Overview
<!-- tags: backend, api, fastapi, routers -->
Date: 2026-01-08
Last Review: 2026-02-06
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
- 2026-01-21: Video tasks now honor Smart flags (`smart_offload`/`smart_fallback`/`smart_cache`) by propagating them into requests and applying `smart_runtime_overrides(...)` inside the video worker thread.
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
