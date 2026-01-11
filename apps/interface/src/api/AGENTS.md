# apps/interface/src/api Overview
<!-- tags: frontend, api, payloads -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2026-01-06
Status: Active

## Purpose
- Typed API client and DTO definitions used by the frontend to interact with the Codex backend.

## Notes
- Keep request/response types synchronized with `.sangoi/backend/interfaces/schemas/`.
- Regenerate or update the client whenever backend schemas change.
- Reference: `.sangoi/reference/models/model-assets-selection-and-inventory.md` is the canonical “how models/assets are listed + selected” doc (inventory → SHA selection → backend resolution).
- `payloads.ts` now carries both `extras.refiner` and nested `extras.highres.refiner`; `HighresOptionsSchema` includes `refiner` and the builder only emits it when enabled.
- `payloads_video.ts` provides typed (Zod) payload builders for WAN `/txt2vid` and `/img2vid`, including stage overrides (`wan_high/wan_low`), normalization of QuickSettings-style TE labels (`wan22/<abs_path>` → `<abs_path>`), and guards against sentinel asset values (`Automatic`/`Built-in`) so video endpoints receive real paths only.
- 2026-01-06: `/api/samplers` DTO is now `{name,supported,default_scheduler,allowed_schedulers}` and WAN payload builders fail fast on non-canonical (uppercase) sampler/scheduler inputs.
- 2025-12-16: Added `startVid2Vid(FormData)` for `/api/vid2vid` (multipart upload) and a typed builder `buildWanVid2VidPayload()`; video task events/results now include an optional `video { rel_path, mime }` export descriptor for `/api/output/{rel_path}`.
- `Txt2ImgRequestSchema` exposes optional `smart_offload`/`smart_fallback` booleans so quicksettings can toggle smart offload and CPU fallback per-generation (mirroring `/api/options` keys `codex_smart_offload`/`codex_smart_fallback`).
- Inventory helpers (`InventoryResponse`) are served by `/api/models/inventory`; the client exposes both a cached fetch (`fetchModelInventory`) and a rescan path (`refreshModelInventory`) that posts to `/api/models/inventory/refresh` (assets like VAEs/Text Encoders/metadata roots).
- `ModelsResponse` is served by `/api/models`; the client exposes a rescan path (`refreshModels`) that calls `/api/models?refresh=1` so the UI can pick up newly copied checkpoints without restarting the backend.
- `payloads.ts` also exposes `deriveFluxTextEncoderOverrideFromLabels(labels)`, which builds a `text_encoder_override` payload (family `flux1`, label `flux1/explicit`, components `clip_l=/abs/...`, `t5xxl=/abs/...`) from Flux.1-style `flux1/<abs_path>` labels; model-tab generation (`useGeneration(tabId)`) reuses this helper when sending Flux.1 txt2img requests.
- 2026-01-02: Added standardized file header blocks to `client.ts` and `payloads.ts` (doc-only change; part of rollout).
- 2026-01-03: Added standardized file header block to `types.ts` (doc-only change; part of rollout).
- 2026-01-04: `payloads.ts` treats Flux.1 family keys as flow engines (`flux1*`) for distilled-CFG handling (legacy engine key aliases are not accepted).
