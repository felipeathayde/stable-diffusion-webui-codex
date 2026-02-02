# apps/backend/runtime/vision/upscalers Overview
<!-- tags: backend, runtime, vision, upscalers, spandrel, tiling -->
Date: 2026-02-01
Owner: Backend Runtime Maintainers
Last Review: 2026-02-01
Status: Active

## Purpose
- Provide a **global upscalers runtime** (standalone and hires-fix) with:
  - strict model discovery from `apps/paths.json` roots (`upscale_models`, `latent_upscale_models`);
  - a unified Spandrel backend (load + run) without leaking Spandrel types into callers;
  - tiled inference with overlap and explicit OOM handling.

## Key Files
- `specs.py` — Typed configs (`TileConfig`, `UpscalerDefinition`, builtin latent upscalers).
- `registry.py` — Local discovery + caching + “run upscale” orchestration.
- `spandrel_backend.py` — The only module allowed to import `spandrel`.
- `tiled_scale.py` — Dependency-light tiled scaling utility (used by upscalers and VAE patcher).

## Invariants
- Fail loud:
  - Unknown upscaler id → raise `UpscalerNotFoundError`.
  - Missing/unsupported weights → raise `UpscalerLoadError` with actionable instructions.
- No silent fallback behaviour:
  - OOM handling is explicit and controlled via `TileConfig.fallback_on_oom`.
  - If fallback is disabled, raise the OOM error with a suggested tile size.
- Keep dependencies narrow:
  - Only `spandrel_backend.py` imports `spandrel`.

