# AGENT — Runtime State Dict Helpers

Purpose: Lightweight state-dict mapping views + small state-dict utilities used by loaders and runtime codepaths.

Key files:
- `apps/backend/runtime/state_dict/tools.py`: Small state-dict utilities and diagnostics helpers.
- `apps/backend/runtime/state_dict/views.py`: Mapping views (prefix/filter/remap/cast) + `LazySafetensorsDict`.

Notes:
- Views should stay lightweight and avoid eagerly materializing large state dicts.
- Helpers should remain generic and not import model-family runtime code.

Last Review: 2026-01-18

