# apps/backend/runtime/kernels Overview
Status: Active

## Purpose
- Owns custom C++/CUDA extensions used by runtime code.

## Active kernel trees
- `wan_fused_v1/` — WAN fused self/cross attention addon sources and build scripts.

## Expectations
- Keep build inputs aligned with the runtime loaders that consume each extension.
- Remove retired kernel trees instead of leaving dormant sources behind.
