# apps/backend/core Overview
Date: 2025-10-28
Owner: Backend Core Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Provides the fundamental building blocks for backend orchestration: device discovery, state tracking, RNG, request handling, engine contracts, and parameter parsing.

## Subdirectories
- `params/` — Typed parameter schemas and helpers (currently video-specific) that translate user requests into engine-friendly structures.

## Key Files
- `engine_interface.py` — Defines the base interfaces that engines must implement.
- `orchestrator.py` — Coordinates use-case execution, binding requests to engines and runtime contexts.
- `devices.py` / `state.py` — Track hardware availability and request-scoped generation state.
- `rng.py` / `philox.py` — Native RNG stack (CPU/GPU/Philox) used across tasks.
- `requests.py` — Typed request objects and validation helpers.
- `registry.py` — Engine registration/lookup for orchestration.
- `sampler_policy.py` — Central policy for sampler selection and validation.
- `progress_tracker.py` / `progress_stream.py` — Shared progress reporting utilities.
- `exceptions.py` — Core exception types surfaced by orchestration.

## Notes
- New engine integrations must conform to `engine_interface.py` and register via `registry.py`.
- Keep RNG and device logic centralized here—avoid duplicating random seeding in downstream modules.
