# AGENT — Model Parser
<!-- tags: runtime, model-parser -->
Date: 2025-10-29
Owner: Runtime Maintainers
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
- Provide synthetic/unit test coverage for parser plans.

## Notes
- 2025-11-04: SDXL UNet converter now normalizes label embedding keys (Forge-style `label_emb.0.0.*`) before load, preventing missing/extra weights for custom checkpoints.

## Notes
- 2025-11-04: Parser execution now materializes lazy safetensor components via dedicated helpers before running converters, preventing repeated file handle churn on Windows.
- 2025-11-23: SDXL CLIP validation now fails fast when `transformer.text_model.encoder.layers.0.layer_norm1.weight` is missing on either encoder, stopping pruned/partial checkpoints before they produce garbage conditioning.
