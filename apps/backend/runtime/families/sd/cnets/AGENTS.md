# apps/backend/runtime/sd/cnets Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2026-01-18
Status: Active

## Purpose
- ControlNet support utilities for Stable Diffusion pipelines (conditioning, module wrappers).

## Notes
- `cldm.py` defines the SD-family ControlNet module backed by shared UNet blocks under `apps/backend/runtime/common/nn/unet/`.
- `t2i_adapter.py` defines the SD-family T2I-Adapter modules; shared conv/pool factories come from `apps/backend/runtime/common/nn/unet/utils.py`.
- Keep this folder SD-family specific; shared helpers belong under `apps/backend/runtime/common/`.
