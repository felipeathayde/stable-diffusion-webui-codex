# Backend Shim Inventory (2025-10-27)

All shims removed — legacy `backend/` package deleted in favor of the unified tree under `apps/backend`.

Status
- Restructure date: 2025-10-27
- Import redirector removed. Old imports `backend.*` fail with `ModuleNotFoundError`.
- Backend entrypoints now live under:
  - `apps/backend/use_cases/`
  - `apps/backend/engines/**`
  - `apps/backend/runtime/**`
  - `apps/backend/infra/**`
  - `apps/backend/interfaces/**`
- The process launcher now lives under `apps/launcher/` (package) with the BIOS TUI at `apps/tui_bios.py`.

Guidance
- Use `apps.backend.*` imports nativamente. Se alguma façade exposta ainda for necessária, exporte via `apps/backend/__init__.py`.
- Evite depender de `apps.server.*` — o namespace foi eliminado; use `apps.backend.*` e `apps.launcher`.
