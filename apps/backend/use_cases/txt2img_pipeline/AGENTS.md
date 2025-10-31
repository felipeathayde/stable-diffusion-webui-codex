# apps/backend/use_cases/txt2img_pipeline Overview
Date: 2025-10-31
Owner: Backend Use-Case Maintainers
Last Review: 2025-10-31
Status: Active

## Purpose
- Hosts the staged txt2img pipeline runner (`Txt2ImgPipelineRunner`) used by `apps/backend/use_cases/txt2img.py`.
- Encapsulates preparation, base sampling, and HiRes passes with explicit dataclasses for intermediate state.

## Notes
- When extending txt2img behaviour (e.g., new stages, telemetry), add dedicated methods or dataclasses here instead of enlarging `txt2img.py`.
- Stage helpers must log meaningful events and raise explicit errors; no silent fallbacks or global state mutations.
