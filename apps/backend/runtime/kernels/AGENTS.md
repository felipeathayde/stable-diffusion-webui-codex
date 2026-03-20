# apps/backend/runtime/kernels Overview
Status: Active

## Purpose
- Owns custom C++/CUDA extensions used by runtime code.

## Kernel Trees
- `attention_sram_v1/` — generic SRAM/shared-memory attention addon sources and build scripts.
- `wan_fused_v1/` — retired WAN-specific prototype awaiting removal after generic cutover.

## Expectations
- Keep build inputs aligned with the runtime loaders that consume each extension.
- Remove retired kernel trees instead of leaving dormant sources behind.
