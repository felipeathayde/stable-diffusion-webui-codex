# Runtime Models — AGENTS Notes
Date: 2025-11-03
Owner: Runtime Maintainers
Status: Active

## Scope
Applies to `apps/backend/runtime/models/*` including `loader.py` and state-dict helpers.

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
