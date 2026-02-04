<!-- tags: backend, runtime, supir, nn, unet, control -->

# apps/backend/runtime/families/supir/nn Overview
Date: 2026-02-02
Last Review: 2026-02-02
Status: In progress

## Purpose
- Host SUPIR-specific neural network modules (Control + UNet adapters) required to run SUPIR Enhance.
- Keep these modules self-contained and reuse Codex-native UNet building blocks where possible (`apps/backend/runtime/common/nn/unet/*`).

## Key files
- `zero.py` — Zero-initialized adapters used to fuse control features (ZeroSFT/ZeroCrossAttn).
- `control.py` — SUPIR control network (GLVControl) producing a list of control tensors.
- `unet.py` — SUPIR UNet variant (LightGLVUNet) that consumes control tensors and produces the noise prediction.

## Notes
- Weight compatibility is strict on structure: module/attribute names must remain stable once SUPIR weights are supported.
- These modules are torch-heavy; keep imports local in routers/tasks (import-light policy).

