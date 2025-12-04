# apps/backend/use_cases/txt2img_pipeline Overview
<!-- tags: backend, use-case, txt2img, pipeline, refiner -->
Date: 2025-10-31
Owner: Backend Use-Case Maintainers
Last Review: 2025-12-03
Status: Active

## Purpose
- Hosts the staged txt2img pipeline runner (`Txt2ImgPipelineRunner`) used by `apps/backend/use_cases/txt2img.py`.
- Encapsulates preparation, base sampling, HiRes, and optional SDXL Refiner passes with explicit dataclasses for intermediate state.

## Notes
- When extending txt2img behaviour (e.g., new stages, telemetry), add dedicated methods or dataclasses here instead of enlarging `txt2img.py`.
- Stage helpers must log meaningful events and raise explicit errors; no silent fallbacks or global state mutations.
- Refiner stages are modelled as `RefinerStage` implementations (`GlobalRefinerStage`, `HiresRefinerStage`) under `refiner.py`; runner composes them explicitly after base/hires sampling.
- SDXL refiner config now travels via typed `RefinerConfig` on `CodexProcessingTxt2Img` (global) and `CodexHighResConfig.refiner` (hires-coupled). Missing `model`/steps raise at stage execution instead of silently skipping.
