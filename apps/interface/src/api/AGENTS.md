# apps/interface/src/api Overview
<!-- tags: frontend, api, payloads -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2026-02-04
Status: Active

## Purpose
- Typed API client and DTO definitions used by the frontend to interact with the Codex backend.

## Notes
- Keep request/response types synchronized with `.sangoi/backend/interfaces/schemas/`.
- Regenerate or update the client whenever backend schemas change.
- Reference: `.sangoi/reference/models/model-assets-selection-and-inventory.md` is the canonical “how models/assets are listed + selected” doc (inventory → SHA selection → backend resolution).
- `payloads.ts` now carries both `extras.refiner` and nested `extras.hires.refiner`; `HiresOptionsSchema` includes `refiner` and the builder only emits it when enabled.
- `payloads_video.ts` provides typed (Zod) payload builders for WAN video endpoints (sha-first): stages use `model_sha` + optional `lora_sha` (sha256), TE/VAE use sha selection, and builders guard against sentinel asset values (`Automatic`/`Built-in`). For `vid2vid.method="wan_animate"`, the backend requires repo-scoped paths (under `CODEX_ROOT`) for `vid2vid_model_dir` and stage `model_dir`; stage LoRA remains sha-only via `lora_sha`.
- 2026-01-23: `payloads_video.ts` snaps WAN video `width/height` up to a multiple of 16 (rounded up; Diffusers parity) so requests never trip backend `%16` validation.
- 2026-01-23: `client.ts` now extracts FastAPI `{"detail": ...}` error bodies into readable `Error.message` strings (no more opaque “400 Bad Request”).
- 2026-01-24: Removed the static `/settings_schema.json` fallback; the frontend now requires `/api/settings/schema` to be available.
- `ModelsResponse` is served by `/api/models`; it includes `core_only` plus `core_only_reason` so UIs can explain why a checkpoint is treated as core-only (suffix remains a fallback).
- `EngineCapabilitiesResponse` is served by `/api/engines/capabilities`; it includes:
  - `asset_contracts` (base + core-only; now includes `tenc_slots`/`tenc_slot_labels` for slot-accurate requirements)
  - `engine_id_to_semantic_engine` (explicit key-space mapping)
- 2026-01-06: `/api/samplers` DTO is now `{name,supported,default_scheduler,allowed_schedulers}` and WAN payload builders fail fast on non-canonical (uppercase) sampler/scheduler inputs.
- 2025-12-16: Added `startVid2Vid(FormData)` for `/api/vid2vid` (multipart upload) and a typed builder `buildWanVid2VidPayload()`; video task events/results now include an optional `video { rel_path, mime }` export descriptor for `/api/output/{rel_path}`.
- 2026-01-27: WAN payload builders now optionally emit `video_return_frames` (default off) to control whether txt2vid/img2vid results include frames (and whether vid2vid returns preview frames); video export remains controlled via `video_save_output`.
- `Txt2ImgRequestSchema` exposes optional `smart_offload`/`smart_fallback` booleans so quicksettings can toggle smart offload and CPU fallback per-generation (mirroring `/api/options` keys `codex_smart_offload`/`codex_smart_fallback`).
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
