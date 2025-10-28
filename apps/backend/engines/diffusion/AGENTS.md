# apps/backend/engines/diffusion Overview
Date: 2025-10-28
Owner: Engine Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Task-layer diffusion runners (txt2img, img2img, txt2vid, img2vid) that connect use cases with model-specific engines.

## Notes
- Keep orchestration logic generic; delegate model-specific behaviour to the respective engine subpackages and runtimes.
