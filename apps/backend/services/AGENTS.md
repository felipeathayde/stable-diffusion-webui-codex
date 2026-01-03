# apps/backend/services Overview
Date: 2025-10-28
Owner: Backend Service Maintainers
Last Review: 2026-01-03
Status: Active

## Purpose
- Encapsulates high-level backend services that expose functionality to the API: media encoding/decoding, options management, progress broadcasting, etc.

## Key Files
- `image_service.py` — Orchestrates image generation requests, batching, and responses.
- `media_service.py` — Handles encode/decode operations and format policies.
- `live_preview_service.py` — Builds per-task live preview config from Settings and attaches encoded preview payloads to progress events.
- `options_store.py` / `options_service.py` — JSON-backed options store (`apps/settings_values.json`) and API-friendly wrapper service.
- `progress_service.py` — Aggregates progress reports emitted by use cases/engines.

## Notes
- Services should remain stateless apart from request-scoped state managed in `core/state.py`.
- When introducing new user-facing capabilities, add service wrappers here and expose them via the API schemas in `apps/backend/interfaces/`.
- 2026-01-01: Centralized live preview Settings parsing + SSE payload encoding/attachment in `live_preview_service.py` to keep API workers thin and avoid duplicating image encoding logic.
- 2025-12-29: `options_service.py` now resolves `apps/settings_values.json` relative to `CODEX_ROOT` (required) so option reads/writes don’t depend on the process CWD.
- 2026-01-03: Added standardized file header docstrings to `services/*` modules (doc-only change; part of rollout).
