# apps/backend/engines/sd Overview
Date: 2025-10-28
Owner: Engine Maintainers
Last Review: 2025-11-01
Status: Active

## Purpose
- Stable Diffusion engine implementations (txt2img/img2img) leveraging the SD runtime components.

## Notes
- Keep SD engine logic aligned with runtime helpers under `runtime/sd/`.
- Shared assembly helpers live in `spec.py` — define engine specs (dataclasses) there and assemble components via `assemble_engine_runtime` inside each engine’s `_build_components` implementation.
- Each engine must expose `EngineCapabilities` (txt2img/img2img) and rely on `_require_runtime()` style guards when touching assembled runtime state.
- Preference order: extend specs first, then consume them in `_build_components`; never reintroduce legacy component dictionaries or silent clip-skip fallbacks.

### Event Emission
- Engines must emit `ProgressEvent` and a final `ResultEvent` for UI/services to render progress and images.
- SDXL `txt2img` decodes latents to RGB and emits a `ResultEvent` with `images` and a JSON `info` string.
- Progress streaming can be added by polling `apps.backend.core.state.state` while sampling or by converting the sampler into an event-yielding generator; keep the approach explicit per engine.
