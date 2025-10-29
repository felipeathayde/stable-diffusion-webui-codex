# Model Registry (Work in Progress)
Date: 2025-10-28
Owner: Backend Maintainers
Status: Draft

## Purpose
- Track structured metadata about supported checkpoints and pipelines.
- Provide detection heuristics and signatures for model loading without relying on `huggingface_guess`.

## Current Status
- Core dataclasses/enums (now `CodexCoreSignature`/`CodexCoreArchitecture`) in place with manifest-driven metadata harvesting.
- Detectors implemented for SD1.x, Flux (dev/schnell), AuraFlow, SD3 / SD3.5 (medium & large families), Stable Cascade (B/C), Wan2.2 (T2V/I2V), Chroma, and Qwen Image.

## TODO
- Add detectors for remaining launch families (KOALA, StableAudio, WAN22 camera/S2V/animate, Chroma Radiance).
- Expose CLI/inspect tooling for diagnostics.
- Wire registry outputs into loader/runtime paths and add regression fixtures.
