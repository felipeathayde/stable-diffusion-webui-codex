# apps/backend/runtime/kernels Overview
Date: 2025-10-28
Last Review: 2026-02-20
Status: Active

## Purpose
- Hosts custom CUDA/C++ kernels required by runtime modules.

## Subdirectories
- `codexpack/` — Build scripts and sources for CodexPack packed GGUF kernels (`torch.ops.codexpack.*`).
- `wan_fused_v1/` — Build scripts and sources for WAN fused attention V1 addon (`torch.ops.wan_fused_v1.{self_fwd,cross_fwd}`).

## Notes
- Document build requirements for each kernel and keep them in sync with runtime loaders.
- 2026-01-02: Added standardized file header docstrings to kernel build scripts (doc-only change; part of rollout).
- 2026-02-20: Removed legacy WAN T5 FP8 kernel sources (`runtime/kernels/wan_t5`) after WAN22 TE runtime was normalized to GGUF-only text-encoder execution.
- 2026-02-25: Added `runtime/kernels/wan_fused_v1` for WAN fused attention V1 CUDA addon (`wan_fused_v1_cuda`) with C++ op registration + CUDA entrypoints aligned to runtime contract wrappers.
