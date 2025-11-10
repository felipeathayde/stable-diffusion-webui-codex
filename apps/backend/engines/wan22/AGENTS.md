# apps/backend/engines/wan22 Overview
Date: 2025-10-28
Owner: Engine Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- WAN22 engine implementations (txt2vid, img2vid, etc.) that coordinate WAN-specific runtime components and GGUF loaders.

## Notes
- Keep WAN engines aligned with `runtime/wan22` and GGUF helpers to ensure strict asset handling.

## Execution Paths
- Diffusers: loads vendor tree and constructs `WanPipeline`; logs device/dtype and component classes (TE/UNet/VAE).
- GGUF: strict assets via `resolve_user_supplied_assets`; text context produced by runtime `wan22.py` without fallbacks.

## Device/Dtype Policy
- CPU only when explicitly requested; otherwise CUDA is required (error if unavailable).
