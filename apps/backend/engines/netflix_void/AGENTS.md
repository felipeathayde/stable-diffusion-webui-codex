# apps/backend/engines/netflix_void Overview
<!-- tags: backend, engines, netflix_void, video, vid2vid -->
Date: 2026-04-03
Last Review: 2026-04-03
Status: Active

## Purpose
- Host the native `netflix_void` engine seam for the VOID vid2vid family.
- Keep `netflix_void` as a thin `BaseVideoEngine` adapter over the dedicated family loader/runtime scaffold.

## Key Files
- `apps/backend/engines/netflix_void/netflix_void.py` — `NetflixVoidEngine` facade (`engine_id="netflix_void"`, `TaskType.VID2VID`).
- `apps/backend/engines/netflix_void/spec.py` — Typed runtime assembly + runtime container for the VOID scaffold.
- `apps/backend/engines/netflix_void/factory.py` — Factory seam returning the loaded `NetflixVoidEngineRuntime`.
- `apps/backend/engines/netflix_void/__init__.py` — Package export surface.

## Notes
- This package is a staging seam, not a fake runnable lane: the native Pass 1 -> warped-noise -> Pass 2 runtime still fails loud until the repo-owned execution port lands.
- Base-bundle resolution and literal Pass 1/Pass 2 pairing live under `apps/backend/runtime/families/netflix_void/**`, not here.
- Canonical task ownership remains in `apps/backend/use_cases/vid2vid.py`.
