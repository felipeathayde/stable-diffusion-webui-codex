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
- 2026-02-25: Streaming kernel memory path was tightened for Ampere-first execution (no Hopper-only intrinsics): removed separate probability-tile tensor materialization (`p`) by reusing score tile in-place, and removed full-tensor terminal cast allocation by writing output directly in destination dtype per chunk.
- 2026-02-25: Kernel now enforces streaming invariants at runtime (fail-loud): caps tile area and score-tile bytes, and rejects long-sequence configurations that would collapse into full-attention tile execution (`q_chunk==q_len` and `kv_chunk==kv_len`).
- 2026-02-25: Self-attention ABI now receives separated projection tensors (`w_q/w_k/w_v` and optional `b_q/b_k/b_v`) to avoid temporary `w_qkv` packing and internal split-copy overhead that inflated VRAM peak.
- 2026-02-25: Extension ABI version set to `WAN_FUSED_V1_ABI=2`; runtime loader rejects stale modules with older ABI and continues fallback/build path.
- 2026-02-25: Self-path bias contract remains all-or-none across q/k/v; mixed presence is rejected fail-loud to preserve old semantics.
- 2026-02-25: Self/cross forward now run chunked projection + RoPE before attention: `q` is produced per query chunk, `k/v` are cached in BHLD chunks, and output projection is written per query chunk. This removes full-sequence `q/k/v` + full `attn_bhld` materialization from the hot path and lowers first-block VRAM peak.
- 2026-02-26: Kernel telemetry is env-gated and emitted from CUDA path to `stderr` with prefix `[wan_fused_v1.trace]`, including `alloc/reserved/free/total/max_alloc/max_reserved` and q/kv chunk coordinates. Controls: `CODEX_WAN_FUSED_V1_KERNEL_TRACE`, `CODEX_WAN_FUSED_V1_KERNEL_TRACE_KV`, `CODEX_WAN_FUSED_V1_KERNEL_TRACE_EVERY_Q`, `CODEX_WAN_FUSED_V1_KERNEL_TRACE_EVERY_KV`.
- 2026-02-26: Attention core mode selector added: `CODEX_WAN_FUSED_V1_ATTN_CORE=aten|cuda_experimental` (default `aten`). `cuda_experimental` boots a first custom CUDA streaming update kernel (no global score tensor materialization) for head_dim `<=128`; unsupported head dims log trace marker and continue on ATen path.
