# apps/backend/runtime/kernels Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2026-01-02
Status: Active

## Purpose
- Hosts custom CUDA/C++ kernels required by runtime modules (e.g., WAN T5 encoder ops).

## Subdirectories
- `wan_t5/` — Build scripts and sources for WAN T5 CUDA kernels.
- `codexpack/` — Build scripts and sources for CodexPack packed GGUF kernels (`torch.ops.codexpack.*`).

## Notes
- Document build requirements for each kernel and keep them in sync with runtime loaders.
- 2026-01-02: Added standardized file header docstrings to kernel build scripts (doc-only change; part of rollout).
