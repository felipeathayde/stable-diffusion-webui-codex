# Backend Shim Inventory (2025-10-27)

All shims removed — legacy `backend/` package deleted in favor of the unified tree under `apps/backend` (with façade compatibility under `apps.server`).

Status
- Restructure date: 2025-10-27
- Import redirector removed. Old imports `backend.*` fail with `ModuleNotFoundError`.
- Backend entrypoints now live under:
  - `apps/backend/use_cases/`
  - `apps/backend/engines/**`
  - `apps/backend/runtime/**`
  - `apps/backend/infra/**`
  - `apps/backend/interfaces/**`
- Compatibility shims retained at `apps/server/run_api.py` and `apps/server/settings_*` to ease transition.

Guidance
- Use `apps.backend.*` imports nativamente. Se alguma façade exposta ainda for necessária, exporte via `apps/backend/__init__.py`.
- Evite depender de `apps.server.*` — mantido apenas como compatibilidade temporária.
