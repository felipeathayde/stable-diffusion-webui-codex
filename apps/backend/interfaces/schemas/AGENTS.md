<!-- tags: backend, schemas, settings -->
# apps/backend/interfaces/schemas Overview
Date: 2026-01-24
Last Review: 2026-01-24
Status: Active

## Purpose
- Host backend-owned JSON schemas and generated registries that define stable UI/API contracts.

## Key Files
- `apps/backend/interfaces/schemas/settings_schema.json` — source-of-truth settings schema (categories/sections/fields) served to the WebUI.
- `apps/backend/interfaces/schemas/settings_registry.py` — generated Python registry used by the backend to serve schema + validate/prune persisted option values.

## Notes
- Do not edit `settings_registry.py` by hand; regenerate it after changing `settings_schema.json`:
  - `CODEX_ROOT=$PWD PYTHONPATH=$PWD python .sangoi/dev/tools/settings/generate_settings_registry.py`
- 2026-01-24: `codex_attention_backend` choices now exclude the unported `sage` option (kept strict: `torch-sdpa|xformers`).
