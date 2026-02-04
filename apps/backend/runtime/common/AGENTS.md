# apps/backend/runtime/common Overview
Date: 2025-10-28
Last Review: 2026-01-25
Status: Active

## Purpose
- Shared runtime utilities available to multiple model families (base layers, helpers).

## Subdirectories
- `nn/` — Common neural network modules (core transformers/UNets, attention blocks, etc.).

## Notes
- Add reusable building blocks here to avoid duplication across model-specific runtimes.
- `vae.py` normalises Flow16 VAE safetensors by stripping common prefixes, reusing SDXL/Flux LDM→diffusers key conversion for Z Image, and fails fast on incompatible (non-16-channel) VAEs to avoid noisy decodes.
  - Flow16 config parity: includes `use_quant_conv=false` / `use_post_quant_conv=false` (HF Flux/Z-Image configs) so missing quant conv weights do not trigger false drift warnings.
- 2026-01-25: Flow16 VAE state_dict normalization now reuses the strict SDXL VAE keymap (`apps/backend/runtime/state_dict/keymap_sdxl_vae.py`), drops `model_ema.decay` / `model_ema.num_updates`, and fails loud on unknown non-weight keys (no silent “skip conversion” path).
- 2026-01-06: `vae.load_flow16_vae(...)` now accepts `.gguf` weights (dequantized upfront) in addition to diffusers directories and `.safetensors` files.
- 2026-01-02: Added standardized file header docstrings to `nn/base.py`, `nn/clip.py`, and `nn/unet/{__init__,config,utils}.py` (doc-only change; part of rollout).
