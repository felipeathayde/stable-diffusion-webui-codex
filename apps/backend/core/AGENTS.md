# apps/backend/core Overview
Date: 2025-10-28
Owner: Backend Core Maintainers
Last Review: 2026-01-04
Status: Active

## Purpose
- Provides the fundamental building blocks for backend orchestration: device discovery, state tracking, RNG, request handling, engine contracts, and parameter parsing.

## Subdirectories
- `params/` — Typed parameter schemas and helpers (currently video-specific) that translate user requests into engine-friendly structures.

## Key Files
- `engine_interface.py` — Defines the base interfaces that engines must implement.
- `orchestrator.py` — Coordinates use-case execution, binding requests to engines and runtime contexts.
- `devices.py` / `state.py` — Track hardware availability and request-scoped generation state.
- `engine_loader.py` — Bundle-aware engine loader used by use cases for model loading and runtime option application.
- `rng.py` / `philox.py` — Native RNG stack (CPU/GPU/Philox) used across tasks.
- `requests.py` — Typed request objects and validation helpers.
- `registry.py` — Engine registration/lookup for orchestration.
- `sampler_policy.py` — Central policy for sampler selection and validation.
- `progress_tracker.py` / `progress_stream.py` — Shared progress reporting utilities.
- `exceptions.py` — Core exception types surfaced by orchestration.

## Notes
- New engine integrations must conform to `engine_interface.py` and register via `registry.py`.
- Keep RNG and device logic centralized here—avoid duplicating random seeding in downstream modules.
- 2025-12-14: Video requests (`Txt2VidRequest`/`Img2VidRequest`) include `steps` explicitly (defaulting to 30 to match `/api/{txt2vid,img2vid}`) so API parsing and `build_video_plan()` stay aligned.
- 2025-12-16: Added `TaskType.VID2VID` + `Vid2VidRequest` for WAN video-to-video orchestration; video requests also carry optional `video_options` for export settings.
- 2025-12-30: `InferenceOrchestrator` now reloads an already-loaded engine when load-affecting `engine_options` change (e.g. `text_encoder_override`, VAE override, core streaming), so overrides actually apply and caches don’t go stale across requests.
- 2026-01-01: `InferenceOrchestrator` now purges VRAM (unload cached engines + memory manager unload/empty_cache) before a generation when the requested `(checkpoint, text encoders)` signature differs from the previous generation (prevents OOM on model swaps).
- 2026-01-02: Added standardized file header docstrings across `apps/backend/core/**` modules (doc-only change; part of rollout).
- 2026-01-03: `apps/backend/core/__init__.py` no longer re-exports star-import facades; callers must import from specific modules (e.g. `core.requests`, `core.registry`).
