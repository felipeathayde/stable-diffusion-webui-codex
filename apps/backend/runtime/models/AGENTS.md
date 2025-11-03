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

## Error Handling
- Missing/Unexpected above thresholds will be escalated by the loader; we do not degrade silently.
- Prefer clear messages naming a few representative keys and the active normalization path.

## Rationale
- Mirrors Comfy’s `clip_text_transformers_convert` behaviour: convert legacy resblocks to `text_model.*` and accept plain `text_model.*` roots by lifting into the active wrapper namespace.
