<!-- tags: backend, runtime, families, layout -->

# apps/backend/runtime/families Overview
Date: 2026-01-17
Last Review: 2026-01-17
Status: Active

## Purpose
- Host model-family runtime code (WAN22/Flux/SD/ZImage/Chroma) under a single `families/` root so `apps/backend/runtime/` stays reserved for generic, cross-family runtime modules (models/loaders, ops, sampling, memory, vision, etc.).

## Structure
- `wan22/`, `flux/`, `sd/`, `zimage/`, `chroma/` — family runtimes (implementation-specific).

## Notes
- Avoid facade-first imports; prefer importing the defining module within each family runtime.
- See the plan: `.sangoi/plans/2026-01-17-backend-runtime-families-layout.md`.

