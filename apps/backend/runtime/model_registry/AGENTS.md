# Model Registry (Work in Progress)
Date: 2025-10-28
Owner: Backend Maintainers
Last Review: 2025-12-14
Status: Draft

## Purpose
- Track structured metadata about supported checkpoints and pipelines.
- Provide detection heuristics and signatures for model loading without relying on `huggingface_guess`.

## Current Status
- Core dataclasses/enums (now `CodexCoreSignature`/`CodexCoreArchitecture`) in place with manifest-driven metadata harvesting.
- Detectors implemented for SD1.x, SDXL (base/refiner), Flux (dev/schnell), AuraFlow, SD3 / SD3.5 (medium & large families), Stable Cascade (B/C), Wan2.2 (T2V/I2V), Chroma, and Qwen Image.
 - `capabilities.py` defines `SemanticEngine` and an `EngineParamSurface` describing which high-level UI parameter sections (txt2img/img2img/video/highres/refiner/LoRA/ControlNet) are expected to be used for each semantic engine tag; exposed to the API for frontend gating.
 - 2025-12-12: Z Image runtime metadata was corrected (`context_dim=2560`, `flow_shift=3.0`) to match the canonical HF assets for Z-Image Turbo.
 - 2025-12-13: Z Image Turbo default steps adjusted to 9 to match diffusers `ZImagePipeline` recommendation (≈8 effective updates; last `dt=0`).
 - 2025-12-14: `ModelFamily.ZIMAGE.flow_shift` re-aligned to `3.0` in `family_runtime.py` (HF scheduler_config parity).

## TODO
- Add detectors for remaining launch families (KOALA, StableAudio, WAN22 camera/S2V/animate, Chroma Radiance).
- Extend Flux detection to cover additional GGUF layouts when they appear; the current `FluxCoreGGUFDetector` targets core-only Flux transformers (double_blocks.+guidance) with external TEnc/VAE.
- Expose CLI/inspect tooling for diagnostics.
- Wire registry outputs into loader/runtime paths and add regression fixtures.
