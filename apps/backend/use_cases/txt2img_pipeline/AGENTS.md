# apps/backend/use_cases/txt2img_pipeline Overview
Date: 2025-10-31
Owner: Backend Use-Case Maintainers
Last Review: 2025-10-31
Status: Active

## Purpose
- Hosts the staged txt2img pipeline runner (`Txt2ImgPipelineRunner`) used by `apps/backend/use_cases/txt2img.py`.
- Encapsulates preparation, base sampling, HiRes, and optional SDXL Refiner passes with explicit dataclasses for intermediate state.

## Notes
- When extending txt2img behaviour (e.g., new stages, telemetry), add dedicated methods or dataclasses here instead of enlarging `txt2img.py`.
- Stage helpers must log meaningful events and raise explicit errors; no silent fallbacks or global state mutations.
- SDXL refiner execution is driven by `extras.refiner` overrides attached to the processing object; failures to resolve the refiner model or build conditioning must raise clear `RuntimeError`s instead of silently skipping the stage.
