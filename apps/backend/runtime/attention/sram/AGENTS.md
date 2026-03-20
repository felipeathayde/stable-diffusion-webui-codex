# apps/backend/runtime/attention/sram Overview
Date: 2026-03-19
Last Review: 2026-03-20
Status: Active

## Purpose
- Generic SRAM/shared-memory attention runtime bridge for versioned CUDA backends.

## Key Files
- `apps/backend/runtime/attention/sram/__init__.py` — Generic mode parsing, retired-env rejection, extension load/build warmup, pre-shaped dispatch attempts, and runtime metrics.

## Notes
- This namespace is generic by design: WAN22 is only the first consumer.
- Keep the active contract on pre-shaped attention tensors (`[B,H,S,D]`); do not reintroduce model-specific projection fusion here.
- Retired WAN-only env keys must fail loud; do not add translation shims.
- 2026-03-19: Bridge-side pre-shaped validation now rejects zero batch/head tuples and mismatched or empty K/V sequence lengths before kernel launch, so unsupported tuples fail at the contract seam instead of surfacing as CUDA-path runtime errors.
- 2026-03-20: Pre-shaped dispatch now accepts non-overlapping dense `[B,H,S,D]` layouts with contiguous head-dim lanes (`stride[-1] == 1`), so WAN22 self-attention can hand the bridge its permuted views without blind Q/K/V materialization. The bridge also mirrors the CUDA causal `q_len <= int32` bound before launch and expects `attn_fwd` to preserve the input layout in its output tensor.
