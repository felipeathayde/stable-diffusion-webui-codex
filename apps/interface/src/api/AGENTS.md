# apps/interface/src/api Overview
<!-- tags: frontend, api, payloads -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-06
Status: Active

## Purpose
- Typed API client and DTO definitions used by the frontend to interact with the Codex backend.

## Notes
- Keep request/response types synchronized with `.sangoi/backend/interfaces/schemas/`.
- Regenerate or update the client whenever backend schemas change.
- `payloads.ts` now carries both `extras.refiner` and nested `extras.highres.refiner`; `HighresOptionsSchema` includes `refiner` and the builder only emits it when enabled.
- `Txt2ImgRequestSchema` exposes optional `smart_offload`/`smart_fallback` booleans so quicksettings can toggle smart offload and CPU fallback per-generation (mirroring `/api/options` keys `codex_smart_offload`/`codex_smart_fallback`).
 - Inventory helpers (`InventoryResponse`) are served by `/api/models/inventory`; the client exposes both a cached fetch (`fetchModelInventory`) and a rescan path (`refreshModelInventory`) that posts to `/api/models/inventory/refresh` so the QuickSettings bar can trigger an explicit filesystem scan when the user hits “Refresh models”.
 - `payloads.ts` also exposes `deriveFluxTextEncoderOverrideFromLabels(labels)`, which builds a `text_encoder_override` payload (family `flux`, label `flux/explicit`, components `clip_l=/abs/...`, `t5xxl=/abs/...`) from the QuickSettings `currentTextEncoders` array for Flux; both the `/flux` view store and `ImageModelTab.vue` reuse this helper when sending Flux txt2img requests.
