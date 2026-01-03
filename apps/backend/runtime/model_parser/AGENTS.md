# AGENT — Model Parser
<!-- tags: runtime, model-parser -->
Date: 2025-10-29
Owner: Runtime Maintainers
Last Review: 2026-01-02
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
- Provide synthetic/unit test coverage for parser plans.

## Notes
- 2025-11-04: SDXL UNet converter now normalizes sequential label-embedding keys (`label_emb.0.0.*`) before load, preventing missing/extra weights for custom checkpoints.

## Notes
- 2025-11-04: Parser execution now materializes lazy safetensor components via dedicated helpers before running converters, preventing repeated file handle churn on Windows.
- 2025-11-28: SDXL CLIP validation is fail-fast: missing essentials (token embeddings, layer_norm1, final LN, text_projection for CLIP-G) raises a `ValidationError` instead of proceeding with partial encoders.
- 2025-12-05: Flux GGUF core-only parser (`families/flux.py`) now registers synthetic text encoder aliases (`clip_l`, `t5xxl`) so `TextEncoderOverrideConfig` can map Flux text encoder roots to `text_encoder`/`text_encoder_2` components even when CLIP/T5 weights live entirely outside the primary checkpoint.
- 2026-01-02: Added standardized file header docstrings to model parser modules (doc-only change; part of rollout).
