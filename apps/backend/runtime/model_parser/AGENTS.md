# AGENT — Model Parser
<!-- tags: runtime, model-parser -->
Date: 2025-10-29
Last Review: 2026-02-28
Status: Draft

## Mandate
- Provide Codex-native parsing of checkpoint state dicts without relying on `huggingface_guess`.
- Split, convert, and validate model components (core transformer/UNet, VAE, text encoders) using registry `ModelSignature` metadata.
- Surface structured `CodexEstimatedConfig` objects for loaders and adapters, including component mappings and quantization hints.

## Structure
- `__init__.py`: public API (`parse_state_dict`).
- `errors.py`: parser-specific exceptions.
- `specs.py`: dataclasses for plans, context, core signatures, and estimated configs.
- `plan.py`: execution engine for declarative parser plans.
- `builders.py`: shared helpers for component registration and config assembly.
- `quantization.py`: quantisation detection/validation helpers.
- `converters/`: component converters (CLIP/T5/VAE).
- `families/`: family-specific planners (SD1.x, SD2.x, SDXL, SD3/SD3.5, Flux, Chroma, ...).

## TODO
- Extend planners to remaining diffusion/video families (e.g., Stable Cascade, Wan camera/HuMo).
- Add explicit plans for GGUF core-only variants (e.g., Flux transformers without embedded text encoders/VAEs) so loaders can compose them with external components.
- For parser-plan changes, run `python3 -m compileall apps/backend/runtime/model_parser` and record manual parser-path validation steps in the task notes.

## Notes
- 2025-11-04: (Historical) SDXL UNet converter introduced normalization for sequential label-embedding keys (`label_emb.0.0.*`) before load.
- 2025-11-04: Parser execution now materializes lazy safetensor components via dedicated helpers before running converters, preventing repeated file handle churn on Windows.
- 2025-11-28: SDXL CLIP validation is fail-fast: missing essentials (token embeddings, layer_norm1, final LN, text_projection for CLIP-G) raises a `ValidationError` instead of proceeding with partial encoders.
- 2025-12-05: Flux GGUF core-only parser (`families/flux.py`) now registers synthetic text encoder aliases (`clip_l`, `t5xxl`) so `TextEncoderOverrideConfig` can map Flux text encoder roots to `text_encoder`/`text_encoder_2` components even when CLIP/T5 weights live entirely outside the primary checkpoint.
- 2026-01-02: Added standardized file header docstrings to model parser modules (doc-only change; part of rollout).
- 2026-01-14: Flux parser now surfaces a targeted error when a GGUF file contains Diffusers Flux keys (`transformer_blocks.*`) instead of the expected Comfy/Codex layout (`double_blocks.*`), directing operators to re-convert via the Tools GGUF converter.
- 2026-01-29: CodexPack `*.codexpack.gguf` now counts as GGUF quantization for parser quantization detection (so loaders build GGUF ops and use hook-based `load_state_dict`, not the conservative copier).
- 2026-02-10: SDXL nested UNet label-embedding key normalization moved out of parser converters into canonical checkpoint keymap (`apps/backend/runtime/state_dict/keymap_sdxl_checkpoint.py`); SDXL parser plan no longer runs UNet key normalization converters.
- 2026-02-10: CLIP converter structural conversions (projection transpose paths in `converters/clip.py`) are now globally policy-gated by `CODEX_WEIGHT_STRUCTURAL_CONVERSION`: `auto` fails loud, `convert` opt-in allows conversion.
- 2026-02-11: CLIP parser converter now delegates to canonical CLIP keymap normalization (`normalize_codex_clip_state_dict_with_layout`) and keeps native layout in AUTO (`qkv_impl=auto`, `projection_orientation=auto`) instead of forcing projection transpose.
- 2026-02-16: WAN parser dispatch/config normalization now handles explicit WAN22 families (`WAN22_5B`, `WAN22_14B`, `WAN22_ANIMATE`) instead of a single `WAN22` family bucket.
- 2026-03-04: Safetensors parser planning now avoids eager root-component tensor reads: split diagnostics use header/source hints (no sample tensor fetch), and quantization detection skips full-value scans for safetensors-backed views.
