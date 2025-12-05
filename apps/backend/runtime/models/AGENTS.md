# Runtime Models — AGENTS Notes
<!-- tags: runtime, models, loader, prediction -->
Date: 2025-12-05
Owner: Runtime Maintainers
Status: Active

## Scope
Applies to `apps/backend/runtime/models/*` including `loader.py`, `registry.py`, and state-dict helpers.

## CLIP (TE) State‑Dict Normalization
- Goal: accept Comfy-style and Diffusers-style layouts without guessing external context.
- Accepted inputs:
  - OpenCLIP legacy: `transformer.resblocks.*` (converted to `transformer.text_model.encoder.layers.*`).
  - Plain modern: `text_model.*` at root (lifted to `transformer.text_model.*`).
  - Aliased: `clip_[lgh].transformer.text_model.*`, `conditioner.embedders.*`, `cond_stage_model.*`, `model.*` — prefixes stripped per-key.
- Policy:
  - Strip known wrappers iteratively per key before conversion.
  - Always attempt Codex converters (`convert_sdxl_clip_*`, fallback to `convert_sd20_clip`, `convert_sd15_clip`); treat success as “essential tensors present”.
  - Lift `text_model.*` into `transformer.text_model.*`, normalize `text_projection` into `transformer.text_projection.weight`, and forward `final_layer_norm.*` similarly.
  - Abort with a `RuntimeError` when essential tensors (`token_embedding`, `position_embedding`, first-layer q_proj, `final_layer_norm`) remain missing after normalization — no silent degradation.

## UNet State‑Dict Normalization
- Accepted inputs:
  - LDM layout: keys already under `input_blocks./middle_block./output_blocks.` — forwarded untouched.
  - Diffusers layout: `conv_in`, `down_blocks.*`, `mid_block.*`, `up_blocks.*`, `time_embedding.*` — converted per config (`num_res_blocks`, `channel_mult`, transformer depths) using the same mapping rules as Comfy (`unet_to_diffusers`).
- Policy:
  - Strip wrappers like `model.diffusion_model.` before inspection.
  - Build diffusers→LDM key map programmatically from the UNet config and remap tensors in place.
  - Preserve optional leftovers (logged at DEBUG) and drop `logit_scale`-style noise.
  - Guard against missing essentials (`input_blocks.0.0.weight`, `time_embed.0.weight`, `out.2.weight`) by raising a `RuntimeError` with representative diffusers keys.

## Error Handling
- Missing/Unexpected above thresholds will be escalated by the loader; we do not degrade silently.
- Prefer clear messages naming a few representative keys and the active normalization path.

## Rationale
- Mirrors Comfy’s `clip_text_transformers_convert` behaviour: convert legacy resblocks to `text_model.*` and accept plain `text_model.*` roots by lifting into the active wrapper namespace.

## Updates
- 2025-11-22: VAE selection now prefers diffusers `AutoencoderKL` for SD/SDXL/Flux/etc., reserving `AutoencoderKLWan` only for WAN22 so SDXL latents are decoded with the proper architecture.
- 2025-11-23: VAE loader now fails fast when weights are missing (e.g., pruned checkpoints without VAE); error names missing key count and asks for a compatible VAE.
- 2025-11-23: VAE loader logs missing/expected/unexpected key counts before raising, making “frame cinza” cases debuggable when a single safetensors lacks VAE tensors.
- 2025-11-23: `_resolve_vae_class` no longer routes non‑WAN22 families through `AutoencoderKLWan` even when the VAE layout looks like LDM; layout is used only for key mapping, not for selecting the WAN22 VAE outside the WAN22 family.
- 2025-11-24: `_maybe_convert_sdxl_vae_state_dict` now materialises lazy SafeTensors views before reshaping mid-attn projections to avoid torch_cpu.dll crashes on Windows during SDXL VAE conversion.
- 2025-11-25: Loader now preserves scheduler-provided `prediction_type` when it disagrees with the model signature, logging the mismatch instead of forcing the signature value; the signature hint remains accessible via `scheduler.config.codex_signature_prediction_type`.
- 2025-12-04: `ModelRegistry` checkpoint discovery agora usa `apps/paths.json["checkpoints"]` como override primário, com fallbacks curados em `models/` (`sd15`, `sdxl`, `flux`) em vez de varrer múltiplas pastas legacy (`stable-diffusion`, `sd`, `checkpoints`).
- 2025-12-05: Text encoder overrides are resolved centrally in `runtime.models.loader` via `TextEncoderOverrideConfig` + `resolve_text_encoder_override_paths`, mapping `(family, /api/text-encoders label, ModelSignature.text_encoders)` to per-component weights. Overrides fail fast when families mismatch, labels are unknown, or expected `<alias>.(safetensors|bin|pt)` files are missing under the configured root.
