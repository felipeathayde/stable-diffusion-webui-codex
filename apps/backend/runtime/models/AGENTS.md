# Runtime Models — AGENTS Notes
<!-- tags: runtime, models, loader, prediction -->
Date: 2025-12-05
Owner: Runtime Maintainers
Last Review: 2026-01-25
Status: Active

## Scope
Applies to `apps/backend/runtime/models/*` including `loader.py`, `registry.py`, and state-dict helpers.

## CLIP (TE) State‑Dict Normalization
- Goal: accept common WebUI-style and Diffusers-style layouts without guessing external context.
- Accepted inputs:
  - OpenCLIP legacy: `transformer.resblocks.*` (converted to `transformer.text_model.encoder.layers.*`).
  - Plain modern: `text_model.*` at root (lifted to `transformer.text_model.*`).
  - Aliased: `clip_[lgh].transformer.text_model.*`, `conditioner.embedders.*`, `cond_stage_model.*`, `model.*` — prefixes stripped per-key.
- Policy:
  - Strip known wrappers iteratively per key before conversion.
  - Always attempt Codex converters (`convert_sdxl_clip_*`, fallback to `convert_sd20_clip`, `convert_sd15_clip`); treat success as “essential tensors present”.
  - Lift `text_model.*` into `transformer.text_model.*`, normalize `text_projection` into `transformer.text_projection.weight`, and forward `final_layer_norm.*` similarly.
  - Drop HF-only buffers (`*.position_ids`) and canonicalize `logit_scale` into the `IntegratedCLIP` keyspace (no `transformer.*` aliases).
  - Abort with a `RuntimeError` when essential tensors (`token_embedding`, `position_embedding`, first-layer q_proj, `final_layer_norm`) remain missing after normalization — no silent degradation.

## UNet State‑Dict Normalization
- Accepted inputs:
  - LDM layout: keys already under `input_blocks./middle_block./output_blocks.` — forwarded untouched.
  - Diffusers layout: `conv_in`, `down_blocks.*`, `mid_block.*`, `up_blocks.*`, `time_embedding.*` — converted per config (`num_res_blocks`, `channel_mult`, transformer depths) using the shared UNet conversion map (`unet_to_diffusers`).
- Policy:
  - Strip wrappers like `model.diffusion_model.` before inspection.
  - Build diffusers→LDM key map programmatically from the UNet config and remap tensors in place.
  - Preserve optional leftovers (logged at DEBUG) and drop `logit_scale`-style noise.
  - Guard against missing essentials (`input_blocks.0.0.weight`, `time_embed.0.weight`, `out.2.weight`) by raising a `RuntimeError` with representative diffusers keys.

## Error Handling
- Missing/Unexpected above thresholds will be escalated by the loader; we do not degrade silently.
- SDXL: UNet/VAE/CLIP loads are strict — any missing/unexpected keys are fatal.
- Prefer clear messages naming a few representative keys and the active normalization path.

## Rationale
- Normalization converges on Diffusers-style text encoder keys: converts legacy resblocks to `text_model.*` and accepts plain `text_model.*` roots by lifting them into the active wrapper namespace.

## Updates
- 2025-11-22: VAE selection now prefers diffusers `AutoencoderKL` for SD/SDXL/Flux/etc., reserving `AutoencoderKLWan` only for WAN22 so SDXL latents are decoded with the proper architecture.
- 2025-11-23: VAE loader now fails fast when weights are missing (e.g., pruned checkpoints without VAE); error names missing key count and asks for a compatible VAE.
- 2025-11-23: VAE loader logs missing/expected/unexpected key counts before raising, making “frame cinza” cases debuggable when a single safetensors lacks VAE tensors.
- 2025-11-23: `_resolve_vae_class` no longer routes non‑WAN22 families through `AutoencoderKLWan` even when the VAE layout looks like LDM; layout is used only for key mapping, not for selecting the WAN22 VAE outside the WAN22 family.
- 2025-11-24: `_maybe_convert_sdxl_vae_state_dict` now materialises lazy SafeTensors views before reshaping mid-attn projections to avoid torch_cpu.dll crashes on Windows during SDXL VAE conversion.
- 2025-12-11: `_maybe_convert_sdxl_vae_state_dict` expanded to cover `ModelFamily.ZIMAGE`, since Z Image uses the same Flow16 VAE layouts as Flux; external VAEs loaded via `runtime.common.vae.load_flow16_vae` reuse this converter.
- 2025-11-25: Loader now preserves scheduler-provided `prediction_type` when it disagrees with the model signature, logging the mismatch instead of forcing the signature value; the signature hint remains accessible via `scheduler.config.codex_signature_prediction_type`.
- 2025-12-04: `ModelRegistry` checkpoint discovery agora usa `apps/paths.json["checkpoints"]` como override primário, com fallbacks curados em `models/` (`sd15`, `sdxl`, `flux`) em vez de varrer múltiplas pastas legacy (`stable-diffusion`, `sd`, `checkpoints`).
- 2025-12-05: Text encoder overrides are resolved centrally by the loader using `TextEncoderOverrideConfig` + `resolve_text_encoder_override_paths` (now in `runtime.models.text_encoder_overrides`), mapping `(family, <family>/<path> label from paths.json, ModelSignature.text_encoders)` to per-component weights. Overrides fail fast when families mismatch, labels are unknown, or expected `<alias>.(safetensors|gguf|bin|pt)` files are missing under the configured root.
- 2025-12-05: Flux GGUF core-only checkpoints (signalled via `ModelSignature.extras["gguf_core_only"]`) now compose with an external VAE resolved from `apps/paths.json["flux1_vae"]`; `_load_flux_vae_state_dict()` scans configured roots for a suitable VAE weights file and fails fast with an explicit error when nothing usable is found, instead of silently running Flux without a VAE.
- 2025-12-06: `TextEncoderOverrideConfig` gained an `explicit_paths` map (`alias -> abs path`) for file-level overrides (e.g., Flux); `resolve_text_encoder_override_paths` supports two modes: explicit path mapping (skipping root lookup) and root-based lookup. In both cases, aliases are validated against `CodexEstimatedConfig.text_encoder_map`, and missing files or unsupported extensions raise `TextEncoderOverrideError` with clear messages.
- 2025-12-30: `apps/backend/runtime/models/__init__.py` switched back to lazy exports (no eager `import safety` / wildcard imports) so `create_api_app` and tests can import the API with a lightweight torch stub.
- 2025-12-30: Text encoder overrides now accept `.gguf` weights; GGUF-packed state dicts are detected so T5 text encoders can load via the `"gguf"` quant path.
- 2026-01-01: `ModelRegistry` checkpoint discovery now lists only file-based weights under `*_ckpt` roots (`.ckpt/.safetensors/.safetensor/.gguf/...`); it no longer treats vendored Hugging Face metadata folders as selectable checkpoints.
- 2026-01-02: `runtime.models.api` gained `find_checkpoint_by_sha(...)` so API layers can resolve checkpoints from short-hash/sha256 identifiers (backed by `models/.hashes.json`).
- 2026-01-04: `ModelRegistry` now exposes public `hash_for(...)` + `flush_hash_cache()` so inventory and other subsystems can request hashes without importing private cache internals.
- 2026-01-06: VAE selection is expressed via engine options (`vae_source` + `vae_path`); the loader does not persist a separate `external_vae_path` metadata key.
- 2026-01-06: Loader now supports `tenc_path` (string or ordered list) as a shorthand for text encoder overrides: paths are mapped onto `ModelSignature.text_encoders` in order and loaded via the existing `TextEncoderOverrideConfig` pipeline (fail-fast on count/alias mismatch).
- 2026-01-06: Refreshed `loader.py` header block to document `tenc_path` shorthand semantics (doc-only change).
- 2026-01-02: Added standardized file header docstrings to `__init__.py`, `api.py`, and `types.py` (doc-only change; part of rollout).
- 2026-01-08: Split state-dict key normalization helpers into `key_normalization.py` and reused them from `loader.py` (UNet remap + transformer prefix stripping).
- 2026-01-08: Moved text-encoder override definitions into `text_encoder_overrides.py`; loader now imports the shared config + resolver from that module.
- 2026-01-14: Flux expected-family loads now use vendored HF metadata to build the signature (selecting `FLUX.1-dev` vs `FLUX.1-schnell` by guidance key presence), avoiding registry detection failures on prefixed Flux checkpoints.
- 2026-01-18: `CheckpointRecord` now includes `core_only`, `core_only_reason` (e.g. `gguf_suffix`, `gguf_magic`), and optional `family_hint`; `/api/models` surfaces these so UIs stop guessing core-only status by suffix alone.
- 2026-01-18: `loader.py` now lazily imports `diffusers`/`transformers` (keeps `create_api_app` import-light for health/models endpoints and torch-stub tests).
- 2026-01-25: SDXL loads are strict on missing/unexpected keys (fail loud); CLIP normalization now drops `position_ids`, canonicalizes `logit_scale`, and keeps only `transformer.text_projection.weight`.
- 2026-01-25: Loader dtype selection no longer overrides memory-manager role defaults using a whole-file SafeTensors “primary dtype” guess; the hint is now debug-only (prevents TE bf16 vs UNet fp16 drift under AUTO).
