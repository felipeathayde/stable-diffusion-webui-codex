# apps/backend/engines/sd Overview
Date: 2025-10-28
Owner: Engine Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Stable Diffusion engine implementations (txt2img/img2img) leveraging the SD runtime components.

## Notes
- Keep SD engine logic aligned with runtime helpers under `runtime/sd/`.
- Shared assembly helpers live in `spec.py` — define engine specs (dataclasses) there and assemble components via `assemble_engine_runtime` to ensure consistent validation/logging across SD variants.
