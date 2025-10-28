# apps/backend/services Overview
Date: 2025-10-28
Owner: Backend Service Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Encapsulates high-level backend services that expose functionality to the API: media encoding/decoding, options management, progress broadcasting, etc.

## Key Files
- `image_service.py` — Orchestrates image generation requests, batching, and responses.
- `media_service.py` — Handles encode/decode operations and format policies.
- `options_service.py` / related modules — Interface with the Codex options store and validation helpers.
- `progress_service.py` — Aggregates progress reports emitted by use cases/engines.

## Notes
- Services should remain stateless apart from request-scoped state managed in `core/state.py`.
- When introducing new user-facing capabilities, add service wrappers here and expose them via the API schemas in `apps/backend/interfaces/`.
