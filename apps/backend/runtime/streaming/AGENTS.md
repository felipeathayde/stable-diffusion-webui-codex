<!-- tags: backend, runtime, streaming, controllers -->

# apps/backend/runtime/streaming Overview
Date: 2026-02-03
Owner: Runtime Maintainers
Last Review: 2026-02-03
Status: Active

## Purpose
- Provide **shared, cross-family** primitives for segment-based core streaming (device placement + eviction + transfer stats).
- Keep Flux/WAN22 streaming controllers in parity without copy/paste drift.

## Scope
- Shared controller logic only (policy enum, transfer stats, controller implementation).
- Family-specific segment/spec/wrapper code stays under `apps/backend/runtime/families/<family>/streaming/`.

## Key files
- `apps/backend/runtime/streaming/controller.py` — shared controller core (policy/stats/controller).

## Notes
- This folder is intentionally small: if a change is family-specific, it does not belong here.
- When updating streaming semantics, ensure both family wrappers keep their public import paths stable:
  - Flux: `apps/backend/runtime/families/flux/streaming/controller.py`
  - WAN22: `apps/backend/runtime/families/wan22/streaming/controller.py`
