# Runtime Models — AGENTS Notes
Date: 2025-11-03
Owner: Runtime Maintainers
Status: Active

## Scope
Applies to `apps/backend/runtime/models/*` including `loader.py` and state-dict helpers.

## CLIP (TE) State‑Dict Normalization
- Goal: accept Comfy-style and Diffusers-style layouts without guessing external context.
- Accepted inputs:
  - OpenCLIP legacy: `transformer.resblocks.*` (converted by converters to `text_model.encoder.layers.*`).
  - Plain modern: `text_model.*` at root (no `transformer.` prefix).
  - Aliased: `clip_[lgh].transformer.text_model.*` (trim + normalize).
- Policy:
  - If any `text_model.*` keys exist, lift them to `transformer.text_model.*` (partial lift, not gated by all-keys).
  - Normalize `text_projection` to `transformer.text_projection.weight` (transpose when required).
  - Preserve `logit_scale` at root (it is initialized by our wrapper; missing is expected).
  - Do not rewrite when keys already live under `transformer.text_model.*`.

## Error Handling
- Missing/Unexpected above thresholds will be escalated by the loader; we do not degrade silently.
- Prefer clear messages naming a few representative keys and the active normalization path.

## Rationale
- Mirrors Comfy’s `clip_text_transformers_convert` behaviour: convert legacy resblocks to `text_model.*` and accept plain `text_model.*` roots by lifting into the active wrapper namespace.
