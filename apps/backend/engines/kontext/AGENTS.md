# apps/backend/engines/kontext Overview
Date: 2025-12-31
Owner: Engine Maintainers
Last Review: 2025-12-31
Status: Active

## Purpose
- Implements the FLUX.1 Kontext engine (Flux-derived image-conditioned flow model).

## Notes
- Kontext reuses the Flux runtime (`apps/backend/engines/flux/spec.py`, `apps/backend/runtime/flux/`) but changes the sampling contract:
  - For img2img, the init image is encoded and supplied as `image_latents` conditioning tokens (not as the starting noisy latent).
- The canonical HF config mirror lives under `apps/backend/huggingface/black-forest-labs/FLUX.1-Kontext-dev`.
