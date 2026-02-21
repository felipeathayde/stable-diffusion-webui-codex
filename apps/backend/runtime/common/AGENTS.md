# apps/backend/runtime/common Overview
Date: 2025-10-28
Last Review: 2026-02-21
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
- 2026-02-11: Global VAE lane policy now treats WAN22 as native-LDM-only: `resolve_vae_layout_lane` fails loud on non-LDM layout or `diffusers_native` override for WAN22 instead of allowing drift between policy and loader behavior.
- 2026-02-16: WAN22 native-LDM-only policy is now enforced across explicit family variants (`WAN22_5B`, `WAN22_14B`, `WAN22_ANIMATE`) via shared `_WAN22_FAMILIES` gate.
- 2026-02-11: `vae_ldm.sanitize_ldm_vae_config` now maps WAN alias config fields (`z_dim`, `base_dim`, `dim_mult`, `num_res_blocks`) into native LDM constructor fields before instantiation, keeping config-to-constructor contracts explicit.
- 2026-02-20: `vae_ldm.py` now owns the full shared native `AutoencoderKL_LDM` implementation (moved out of `runtime/families/wan22/vae.py`); `runtime/families/wan22/vae.py` remains a compatibility re-export shim and `vae_codex3d.py` imports `DiagonalGaussianDistribution` from `runtime/common/vae_ldm.py`.
- 2026-02-11: Added shared native temporal VAE lane module `vae_codex3d.py` (`AutoencoderCodex3D`) with strict diffusers→codex key remap (`remap_codex3d_vae_state_dict`) and config normalization (`sanitize_codex3d_vae_config`) for no-flatten 3D runtime paths.
- 2026-02-16: `vae_codex3d.py` now delegates WAN22 3D VAE key remap ownership to `apps/backend/runtime/state_dict/keymap_wan22_vae.py` to keep model keymaps centralized in `runtime/state_dict`.
- 2026-02-17: `vae_codex3d.py` now mirrors upstream WAN temporal cache semantics in native code (causal conv cache, chunked encode/decode loop, cached 3D upsample handling, nearest-exact upsample), closing WAN2.2 14B I2V decode parity drift without switching to Diffusers model classes.
- 2026-02-21: `vae_codex3d.py` now reduces temporal copy churn by accumulating encode/decode chunks in lists (single final `torch.cat` instead of per-iteration self-cat) and by caching only the last temporal slice in downsample cache paths instead of cloning full feature tensors.
- 2026-01-06: `vae.load_flow16_vae(...)` now accepts `.gguf` weights (dequantized upfront) in addition to diffusers directories and `.safetensors` files.
- 2026-02-15: `vae.load_flow16_vae(...)` now routes GGUF and torch-file state-dict loading through the requested `device` to keep checkpoint placement consistent with runtime target-device selection.
- 2026-01-02: Added standardized file header docstrings to `nn/base.py`, `nn/clip.py`, and `nn/unet/{__init__,config,utils}.py` (doc-only change; part of rollout).
