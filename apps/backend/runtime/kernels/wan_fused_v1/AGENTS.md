# apps/backend/runtime/kernels/wan_fused_v1 Overview
Date: 2026-02-25
Last Review: 2026-02-25
Status: Active

## Purpose
- Build sources for WAN fused attention V1 CUDA addon (`wan_fused_v1_cuda`).
- Registers `torch.ops.wan_fused_v1.self_fwd` and `torch.ops.wan_fused_v1.cross_fwd` used by runtime contract wrappers.

## Key Files
- `apps/backend/runtime/kernels/wan_fused_v1/setup.py` — CUDA extension build script.
- `apps/backend/runtime/kernels/wan_fused_v1/wan_fused_v1_binding.cpp` — Torch op registration and CPU/CUDA dispatch wiring.
- `apps/backend/runtime/kernels/wan_fused_v1/wan_fused_v1_kernels.cu` — CUDA implementations for V1 self/cross fused forward paths.

## Notes
- Module name must remain `wan_fused_v1_cuda` to match runtime loader expectations.
- CPU path must fail loud; CUDA path is required.
- V1 contract scope: inference-only (`dropout=0`), fp16/bf16/fp32, and mandatory RoPE tensors for self/cross paths as enforced in runtime wrapper.
- 2026-02-25: v1.1 attention core replaced global `LxL` score/probability materialization with streaming tiled attention (online softmax accumulator) in `wan_fused_v1_kernels.cu` for both self and cross paths; chunk sizes are tunable via `CODEX_WAN_FUSED_V1_Q_CHUNK` and `CODEX_WAN_FUSED_V1_KV_CHUNK`, and kernel-side parsing is strict/hard-capped (`Q<=512`, `KV<=1024`) with force-mode fail-loud semantics.
