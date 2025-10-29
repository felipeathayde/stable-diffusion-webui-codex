# Model Registry (Work in Progress)
Date: 2025-10-28
Owner: Backend Maintainers
Status: Draft

## Purpose
- Track structured metadata about supported checkpoints and pipelines.
- Provide detection heuristics and signatures for model loading without relying on `huggingface_guess`.

## Current Status
- Core dataclasses/enums in place with manifest-driven metadata harvesting.
- Detectors implemented for SD1.x, Flux (dev/schnell), AuraFlow, SD3, Stable Cascade (B/C) and Wan2.2 (T2V/I2V).

## TODO
- Add detectors for remaining launch families (Chroma, Qwen Image, KOALA, WAN22 camera/S2V/animate).
- Expose CLI/inspect tooling for diagnostics.
- Wire registry outputs into loader/runtime paths and add regression fixtures.
