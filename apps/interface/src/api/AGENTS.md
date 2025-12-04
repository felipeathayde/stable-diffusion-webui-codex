# apps/interface/src/api Overview
<!-- tags: frontend, api, payloads -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-03
Status: Active

## Purpose
- Typed API client and DTO definitions used by the frontend to interact with the Codex backend.

## Notes
- Keep request/response types synchronized with `.sangoi/backend/interfaces/schemas/`.
- Regenerate or update the client whenever backend schemas change.
- `payloads.ts` now carries both `extras.refiner` and nested `extras.highres.refiner`; `HighresOptionsSchema` includes `refiner` and the builder only emits it when enabled.
