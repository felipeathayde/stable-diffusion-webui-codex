# apps/backend/runtime/kernels Overview
Date: 2025-10-28
Last Review: 2026-02-20
Status: Active

## Purpose
- Hosts custom CUDA/C++ kernels required by runtime modules.

## Subdirectories
- `codexpack/` — Build scripts and sources for CodexPack packed GGUF kernels (`torch.ops.codexpack.*`).

## Notes
- Document build requirements for each kernel and keep them in sync with runtime loaders.
- 2026-01-02: Added standardized file header docstrings to kernel build scripts (doc-only change; part of rollout).
- 2026-02-20: Removed legacy WAN T5 FP8 kernel sources (`runtime/kernels/wan_t5`) after WAN22 TE runtime was normalized to GGUF-only text-encoder execution.
