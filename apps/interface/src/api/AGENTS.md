# apps/interface/src/api Overview
<!-- tags: frontend, api, payloads -->
Date: 2025-10-28
Last Review: 2026-02-28
Status: Active

## Purpose
- Typed API client and DTO definitions used by the frontend to interact with the Codex backend.

## Notes
- Keep request/response types synchronized with `.sangoi/backend/interfaces/schemas/`.
- Regenerate or update the client whenever backend schemas change.
- Reference: `.sangoi/reference/models/model-assets-selection-and-inventory.md` is the canonical “how models/assets are listed + selected” doc (inventory → SHA selection → backend resolution).
- `payloads.ts` now carries both `extras.refiner` and nested `extras.hires.refiner`; `HiresOptionsSchema` includes `refiner` and the builder only emits it when enabled.
- 2026-02-08: swap-model payload semantics now use `switch_at_step` (not `steps`) in both global and hires nested refiner payloads; frontend form state uses `swapAtStep`.
- `payloads_video.ts` provides typed (Zod) payload builders for WAN video endpoints (sha-first): stages use `model_sha` + optional `lora_sha` (sha256), TE/VAE use sha selection, and builders guard against sentinel asset values (`Automatic`/`Built-in`). For `vid2vid.method="wan_animate"`, the backend requires repo-scoped paths (under `CODEX_ROOT`) for `vid2vid_model_dir` and stage `model_dir`; stage LoRA remains sha-only via `lora_sha`.
- 2026-01-23: `payloads_video.ts` snaps WAN video `width/height` up to a multiple of 16 (rounded up; Diffusers parity) so requests never trip backend `%16` validation.
- 2026-01-23: `client.ts` now extracts FastAPI `{"detail": ...}` error bodies into readable `Error.message` strings (no more opaque “400 Bad Request”).
- 2026-01-24: Removed the static `/settings_schema.json` fallback; the frontend now requires `/api/settings/schema` to be available.
- `ModelsResponse` is served by `/api/models`; it includes `core_only` plus `core_only_reason` so UIs can explain why a checkpoint is treated as core-only (suffix remains a fallback).
- `EngineCapabilitiesResponse` is served by `/api/engines/capabilities`; it includes:
  - `asset_contracts` (base + core-only; now includes `tenc_slots`/`tenc_slot_labels` for slot-accurate requirements)
  - `engine_id_to_semantic_engine` (explicit key-space mapping; required by frontend taxonomy resolution)
  - `dependency_checks` (backend-owned readiness rows per semantic engine; strict `ready === all(check.ok)` contract)
- 2026-01-06: `/api/samplers` DTO is now `{name,supported,default_scheduler,allowed_schedulers}` and WAN payload builders fail fast on non-canonical (uppercase) sampler/scheduler inputs.
- 2025-12-16: WAN video client helpers include task-event/result handling with optional `video { rel_path, mime }` export descriptor for `/api/output/{rel_path}`.
- 2026-01-27: WAN payload builders now optionally emit `video_return_frames` (default off) to control whether txt2vid/img2vid results include frames (and whether vid2vid returns preview frames); video export remains controlled via `video_save_output`.
- 2026-02-15: Image/video payload contracts now always include `settings_revision`; per-request `smart_offload`/`smart_fallback`/`smart_cache` fields were removed (runtime flags remain `/api/options`-owned).
- 2026-02-15: `client.ts` now caches `/api/options` revision and preserves structured HTTP error payloads (`status/detail/body`) so composables can handle stale-revision `409` conflicts.
- 2026-02-16: `payloads_video.ts` no longer emits WAN scheduler overrides (`txt2vid_scheduler`/`img2vid_scheduler`/`vid2vid_scheduler` nor stage `scheduler`), aligning frontend payloads with WAN22 backend policy (scheduler is runtime-managed).
- 2026-02-17: `payloads_video.ts` now normalizes WAN frame counts to the `4n+1` domain within `[9,401]`, emits `gguf_attention_mode` (`global|sliding`), and supports img2vid chunk controls (`img2vid_chunk_frames`, `img2vid_overlap_frames`, `img2vid_anchor_alpha`, `img2vid_chunk_seed_mode`).
- 2026-02-21: `payloads_video.ts` stage payload schema now accepts stage-scoped prompt fields (`wan_high.prompt/negative_prompt`, `wan_low.prompt/negative_prompt`); top-level mode prompt fields are derived from the High stage prompt at build time (fail-loud when High prompt is empty).
- 2026-02-21: `payloads_video.ts` img2vid temporal contract now requires `img2vid_mode` (`solo|chunk|sliding`), with mode-scoped validation for chunk fields (`img2vid_chunk_*`) versus sliding-window fields (`img2vid_window_frames/stride/commit_frames`).
- 2026-02-22: `payloads_video.ts` now supports `img2vid_mode='svi2'|'svi2_pro'` with the same windowed contract as sliding; temporal normalization is centralized in `utils/wan_img2vid_temporal.ts` (`stride % 4 == 0`, `commit >= stride + 4`).
- 2026-02-22: `payloads_video.ts` now includes optional `img2vid_reset_anchor_to_base` for img2vid temporal modes, allows it in `chunk|sliding`, and enforces fail-loud `false` for `svi2|svi2_pro`; builders map `WanImg2VidInput.resetAnchorToBase` directly to payload field.
- 2026-02-27: `payloads_video.ts` WAN output contract now hard-sets `video_save_output=true` and `video_save_metadata=true`, removed obsolete output fields (`video_filename_prefix`, `video_trim_to_audio`), and maps interpolation through one `targetFps` input (`0` disables; values above base FPS emit `video_interpolation.times >= 2` via `ceil(targetFps/baseFps)` with fixed model `rife47.pth`).
- 2026-02-27: `payloads_video.ts` now includes optional strict `video_upscaling` payload mapping (SeedVR2 fields) via `WanVideoUpscalingInput`; builders emit `video_upscaling` only when enabled, and schema enforces typed ranges/enums plus `batch_size` `4n+1`.
- 2026-02-22: `client.ts` adds `fetchObliterateVram(payload?)` for `POST /api/obliterate-vram` with default `external_kill_mode='disabled'`; `types.ts` defines request/response DTOs (including external kill mode + skip/failure rows) so quick settings can report safe-default cleanup status fail-loud.
- Inventory helpers (`InventoryResponse`) are served by `/api/models/inventory`; the client exposes both a cached fetch (`fetchModelInventory`) and a rescan path (`refreshModelInventory`) that posts to `/api/models/inventory/refresh` (assets like VAEs/Text Encoders/metadata roots).
- `ModelsResponse` is served by `/api/models`; the client exposes a rescan path (`refreshModels`) that calls `/api/models?refresh=1` so the UI can pick up newly copied checkpoints without restarting the backend.
- 2026-01-13: Added `fetchCheckpointMetadata()` for `/api/models/checkpoint-metadata` so the metadata modal payload can be fetched without constructing `file.*` keys client-side.
- Flux.1 generation requests should not send `extras.text_encoder_override`; the backend derives any needed TE override from `extras.tenc_sha` (sha-only selection).
- 2026-01-02: Added standardized file header blocks to `client.ts` and `payloads.ts` (doc-only change; part of rollout).
- 2026-01-03: Added standardized file header block to `types.ts` (doc-only change; part of rollout).
- 2026-01-04: `payloads.ts` treats Flux.1 family keys as flow engines (`flux1*`) for distilled-CFG handling (legacy engine key aliases are not accepted).
- 2026-01-25: `payloads.ts` now allows `clip_skip` in `[0..12]` (0 = “use default”) and sends `clip_skip=0` when selected so the backend can reset clip skip per request without a separate UI toggle.
- 2026-01-28: `payloads.ts` supports Z-Image Turbo/Base by emitting `extras.zimage_variant="turbo"|"base"`; both variants use classic CFG (`cfg` + optional `negative_prompt`) and the variant only affects scheduler semantics (shift=3.0/6.0) and UI defaults.
- 2026-01-29: Added `analyzePngInfo()` client wrapper + `PngInfoAnalyzeResponse` for `POST /api/tools/pnginfo/analyze` (used by the PNG Info view).
- 2026-02-01: Added upscalers client wrappers + DTOs for the standalone upscale surface (`GET /api/upscalers`, `GET /api/upscalers/remote`, `POST /api/upscalers/download`, `POST /api/upscale`).
- 2026-02-03: `/api/upscalers/remote` DTO now includes `manifest_errors` and categorizes `weights[]` as either curated (`curated=true` with `meta`) or raw listing (`curated=false`).
- 2026-02-04: `payloads.ts` now propagates the global hires `min_tile` preference into `extras.hires.tile.min_tile` (clamped to `tile`) to keep hires tile fallback behavior configurable and drift-free.
- 2026-02-05: `ApiTab.type` now includes `anima` in `types.ts` so UI tab persistence contracts match backend tab allowlist updates.
- 2026-02-08: `payloads.ts` now falls back `extras.hires.{prompt,negative_prompt}` to base prompts when hires prompt overrides are blank.
- 2026-02-18: `types.ts` `EngineCapabilities` now includes optional `guidance_advanced` (`GuidanceAdvancedCapabilities`) so image tabs can render CFG Advanced/APG controls strictly from backend capability contract.
- 2026-02-21: `client.ts` no longer exports `startVid2Vid`; frontend WAN generation dispatch is restricted to `startTxt2Vid`/`startImg2Vid` while backend vid2vid remains disabled (501).
