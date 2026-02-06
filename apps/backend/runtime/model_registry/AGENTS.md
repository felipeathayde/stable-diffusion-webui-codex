# Model Registry (Work in Progress)
Date: 2025-10-28
Last Review: 2026-02-06
Status: Draft

## Purpose
- Track structured metadata about supported checkpoints and pipelines.
- Provide detection heuristics and signatures for model loading without relying on `huggingface_guess`.

## Current Status
- Core dataclasses/enums (now `CodexCoreSignature`/`CodexCoreArchitecture`) in place with manifest-driven metadata harvesting.
- Detectors implemented for SD1.x, SDXL (base/refiner), Flux.1 (dev/schnell), AuraFlow, SD3 / SD3.5 (medium & large families), Stable Cascade (B/C), Wan2.2 (T2V/I2V), Chroma, Qwen Image, and Anima (Cosmos Predict2 core `net.*` format).
- `capabilities.py` defines `SemanticEngine` and an `EngineParamSurface` describing which high-level UI parameter sections (txt2img/img2img/video/hires/refiner/LoRA/ControlNet) are expected to be used for each semantic engine tag; exposed to the API for frontend gating.
- 2025-12-14: `ModelSignature` gained a legacy `unet` alias for `core`, keeping tests and older call sites working while the new contract stays `signature.core`.
- 2025-12-14: Qwen Image detector reintroduced (`detectors/qwen_image.py`) and enums extended (`ModelFamily.QWEN_IMAGE`, `LatentFormat.QWEN_IMAGE`).
- 2025-12-12: Z Image runtime metadata was corrected (`context_dim=2560`, `flow_shift=3.0`) to match the canonical HF assets for Z-Image Turbo.
- 2025-12-13: Z Image Turbo default steps adjusted to 9 to match diffusers `ZImagePipeline` recommendation (≈8 effective updates; last `dt=0`).
- 2025-12-14: `ModelFamily.ZIMAGE.flow_shift` re-aligned to `3.0` in `family_runtime.py` (HF scheduler_config parity).
- 2026-01-28: Z Image semantic surface now declares `supports_img2img=true`; Z-Image Turbo/Base flow shift is treated as variant-specific (`shift=3.0` / `shift=6.0`) and resolved from vendored diffusers scheduler configs.
- 2026-01-06: Engine capability surfaces now default to model_index-derived sampler/scheduler pairs (SD15 `pndm`/`ddim`, SDXL `euler`/`euler_discrete`, WAN22 `uni-pc`/`simple`, Hunyuan `ddpm`/`beta`).
- 2026-01-08: Added `flow_shift.py` as the canonical flow-shift resolver from diffusers `scheduler_config.json` (fixed + dynamic) and removed hard-coded `flow_shift` values from family runtime specs where the value is not a true family invariant (Flux/WAN22).
- 2026-01-08: Refreshed file header blocks for `capabilities.py` and `flow_shift.py` to keep the Symbols lists in sync (doc-only change).
- 2026-01-18: Semantic engine surface for `chroma` now declares `supports_img2img=true` to match the registered `flux1_chroma` engine task surface.
- 2026-02-06: `SemanticEngine.ANIMA` capability surface now exposes `supports_txt2img=true` and `supports_img2img=true` after conditioning payload port (`crossattn` + pooled `vector` + `t5xxl_ids/t5xxl_weights`) and compile/sampler pass-through validation.

## TODO
- Add detectors for remaining launch families (KOALA, StableAudio, WAN22 camera/S2V/animate, Chroma Radiance).
- Extend Flux.1 detection to cover additional GGUF layouts when they appear; the current `FluxCoreGGUFDetector` targets core-only Flux.1 transformers (double_blocks.+guidance) with external TEnc/VAE.
- Expose CLI/inspect tooling for diagnostics.
- Wire registry outputs into loader/runtime paths and add regression fixtures.
- 2026-01-02: Added standardized file header docstrings to model registry modules and detectors (doc-only change; part of rollout).
- 2026-01-06: Updated Flux sampler allow-lists in `capabilities.py` to use canonical `SamplerKind` strings (spaces/`++` preserved).
