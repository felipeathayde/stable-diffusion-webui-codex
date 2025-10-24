# Backend Shim Inventory (2025-10-24)

All shims removed — legacy `backend/` package deleted in favor of the unified façade `apps.server.backend`.

Status
- Removal date: 2025-10-24
- Import redirector removed. Old imports `backend.*` will fail with `ModuleNotFoundError`.
- Hugging Face local assets were relocated to `apps/server/backend/huggingface/`.

Guidance
- Use `apps.server.backend.*` imports exclusively. If a façade export is missing, add it to `apps/server/backend/__init__.py` instead of importing deep internals.
