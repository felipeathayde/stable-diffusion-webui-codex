# apps/backend/runtime/models Overview
Date: 2025-10-29
Owner: Runtime Maintainers
Last Review: 2025-11-01
Status: Active

## Purpose
- Model registry, loader helpers, and metadata utilities (checkpoints, VAEs, text encoders) shared across engines.

## Notes
- Keep loader logic centralized here (safe loading, dtype inference) and expose only typed interfaces to engines/use cases.
- `loader.py` now emits `DiffusionModelBundle` artifacts; SD families resolve through `resolve_diffusion_bundle` so engines never touch legacy loaders directly.
- SD3/SD3.5 signatures still drive Hugging Face repo choice and expose `codex_signature`/`codex_variant` on the estimated config.
- Core architecture metadata (`CodexCoreSignature`) drives dtype/offload decisions and trace labels, replacing generic UNet terminology.
