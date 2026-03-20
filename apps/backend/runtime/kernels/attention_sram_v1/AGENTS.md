# apps/backend/runtime/kernels/attention_sram_v1 Overview
Date: 2026-03-19
Last Review: 2026-03-20
Status: Active

## Purpose
- Owns the generic `attention_sram_v1_cuda` extension used by `runtime/attention/sram`.

## Key Files
- `apps/backend/runtime/kernels/attention_sram_v1/setup.py` — local CUDAExtension build script for `attention_sram_v1_cuda`.
- `apps/backend/runtime/kernels/attention_sram_v1/attention_sram_v1_binding.cpp` — ABI export, op registration, and CPU fail-loud stubs.
- `apps/backend/runtime/kernels/attention_sram_v1/attention_sram_v1_kernels.cu` — narrow shared-memory attention forward kernel for pre-shaped `[B,H,S,D]` tensors plus optional generic RoPE helper.

## Notes
- Keep this tree generic: no WAN-only naming, no fake backend selectors, no projection/norm/out-proj logic here.
- The active `attn_fwd` contract is narrow on purpose: CUDA, fp16, non-overlapping dense `[B,H,S,D]` with contiguous head-dim lanes (`stride[-1] == 1`), `head_dim=128`, boolean `is_causal`, and output layout preserved from `q`.
- `rope_blhd_` is optional scaffolding and must stay separate from the `attn_fwd` hot path.
- 2026-03-19: Kernel-side validation now mirrors the bridge on K/V sequence-length agreement, and the tile loop uses explicit `std::min(...)` selection so the first cut stays compileable under the narrow CUDA contract.
- 2026-03-20: The CUDA path now consumes stride-based `[B,H,S,D]` views directly and writes the output with the input layout, instead of forcing caller-side Q/K/V materialization just to satisfy the first SRAM cut.
