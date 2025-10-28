# apps/backend/interfaces Overview
Date: 2025-10-28
Owner: Backend API Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Defines API-facing schemas and adapters that expose backend capabilities to the Codex frontend and external clients.

## Subdirectories
- `api/` — FastAPI endpoint implementations and adapters.
- `schemas/` — Pydantic/dataclass schemas describing request/response payloads.

## Notes
- Keep schemas in sync with the frontend API client (`apps/interface/src/api`).
- Avoid embedding business logic here—delegate to services/use cases and focus on validation and serialization.
