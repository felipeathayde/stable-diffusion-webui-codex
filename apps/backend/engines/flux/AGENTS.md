# apps/backend/engines/flux Overview
Date: 2025-10-28
Owner: Engine Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Flux engine implementations and task wiring leveraging the Flux runtime.

## Notes
- Ensure scheduler and runtime dependencies stay in sync with `apps/backend/runtime/flux/`.
- Shared assembly helpers live in `spec.py`; extend specs there and reuse them across Flux-like engines (Flux, Chroma distilled variants).
