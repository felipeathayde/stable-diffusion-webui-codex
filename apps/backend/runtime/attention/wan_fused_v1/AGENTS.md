# apps/backend/runtime/attention/wan_fused_v1 Overview
Date: 2026-02-25
Last Review: 2026-02-25
Status: Active

## Purpose
- Defines WAN fused-attention V1 contract helpers and runtime bridge points for optional CUDA fused kernels.
- Centralizes fail-loud validation for WAN fused self/cross attention request tuples.

## Key Files
- `apps/backend/runtime/attention/wan_fused_v1/__init__.py` — Runtime contract validator, mode resolver, and extension loader bridge (`prebuilt` -> `in_place` -> optional `jit`).
- `apps/backend/runtime/kernels/wan_fused_v1/wan_fused_v1_binding.cpp` — `torch.ops.wan_fused_v1.{self_fwd,cross_fwd}` registration and CPU/CUDA dispatch wiring.
- `apps/backend/runtime/kernels/wan_fused_v1/wan_fused_v1_kernels.cu` — CUDA entrypoints implementing self/cross V1 fused forward paths.

## Notes
- V1 scope is inference-only (`dropout=0`) and CUDA-only.
- Cross-attention in V1 requires RoPE on Q+K when fused path is enabled.
- Forced mode must fail loud on unsupported tuples or missing extension/kernel ops.
- Non-forced mode may return explicit reason codes and allow caller-level fallback.
