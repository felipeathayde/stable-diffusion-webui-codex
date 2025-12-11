# apps/backend/runtime/common Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Shared runtime utilities available to multiple model families (base layers, helpers).

## Subdirectories
- `nn/` — Common neural network modules (core transformers/UNets, attention blocks, etc.).

## Notes
- Add reusable building blocks here to avoid duplication across model-specific runtimes.
- `vae.py` normalises Flow16 VAE safetensors by stripping common prefixes and fails fast on incompatible (non‑16‑channel) VAEs to avoid noisy decodes.
