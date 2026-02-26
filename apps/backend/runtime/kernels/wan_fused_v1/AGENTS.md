# apps/backend/runtime/kernels/wan_fused_v1 Overview
Date: 2026-02-25
Last Review: 2026-02-26
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
- 2026-02-26: Extension ABI version set to `WAN_FUSED_V1_ABI=4` after updating the weight/bias tensor contract to eliminate wrapper-side transpose/contiguous materialization. Runtime loader rejects stale modules with older ABI and continues fallback/build path.
- 2026-02-26: Weight/bias contract is `nn.Linear`-native: `w_q/w_k/w_v/w_out` are 2D `[out,in]` tensors and biases are 1D `[out]` tensors; kernels use transpose views internally for matmul.
- 2026-02-25: Self-path bias contract remains all-or-none across q/k/v; mixed presence is rejected fail-loud to preserve old semantics.
- 2026-02-25: Self/cross forward now run chunked projection + RoPE before attention: `q` is produced per query chunk, `k/v` are cached in BHLD chunks, and output projection is written per query chunk. This removes full-sequence `q/k/v` + full `attn_bhld` materialization from the hot path and lowers first-block VRAM peak.
- 2026-02-26: Kernel telemetry is env-gated and emitted from CUDA path to `stderr` with prefix `[wan_fused_v1.trace]`, including `alloc/reserved/free/total/max_alloc/max_reserved` and q/kv chunk coordinates. Controls: `CODEX_WAN_FUSED_V1_KERNEL_TRACE`, `CODEX_WAN_FUSED_V1_KERNEL_TRACE_KV`, `CODEX_WAN_FUSED_V1_KERNEL_TRACE_EVERY_Q`, `CODEX_WAN_FUSED_V1_KERNEL_TRACE_EVERY_KV`.
- 2026-02-26: Attention core selector remains `CODEX_WAN_FUSED_V1_ATTN_CORE=aten|cuda_experimental` (`cuda` alias). Effective-core telemetry is resolver-owned and emitted as `(attn_core, attn_core_source, attn_core_raw)`, with source tokens `env|force_default|kernel_default`.
- 2026-02-26: `cuda_experimental` boots a custom CUDA streaming update kernel (no global score tensor materialization) for head_dim `<=128`; unsupported head dims log trace marker and continue on ATen path.
- 2026-02-26: WAN22 model/run now plumb resolver output directly (`attn_core`, `attn_core_source`, `attn_core_raw`) and do not mutate env in the model/run hot path.
- 2026-02-26: Worker1 cache-guard tuning is fail-loud: tune `CODEX_WAN_FUSED_V1_Q_CHUNK` (`1..512`) and `CODEX_WAN_FUSED_V1_KV_CHUNK` (`1..1024`); invalid values or guard violations raise immediately.

## Required Env Matrix (current)
- Force mode + performance (recommended): `CODEX_WAN22_FUSED_ATTN_V1_MODE=force`; optionally pin `CODEX_WAN_FUSED_V1_ATTN_CORE=cuda_experimental` (resolver default in force mode is already `cuda_experimental` with source `force_default`).
- Auto mode (caller fallback allowed): `CODEX_WAN22_FUSED_ATTN_V1_MODE=auto`; `CODEX_WAN_FUSED_V1_ATTN_CORE` optional (`aten` default).
- Fused off: `CODEX_WAN22_FUSED_ATTN_V1_MODE=off` (leave `CODEX_WAN_FUSED_V1_ATTN_CORE` unset).
- Fail-loud cache/streaming guard knobs: `CODEX_WAN_FUSED_V1_Q_CHUNK` (`1..512`) and `CODEX_WAN_FUSED_V1_KV_CHUNK` (`1..1024`).

## Slow-Path Troubleshooting (force mode)
- If force mode is slow, first verify runtime summary fields `attn_core` / `attn_core_source` and ensure core is not explicitly pinned to `aten`.
- Enable `CODEX_WAN_FUSED_V1_KERNEL_TRACE=1` and check phase markers:
  - `aten_long_seq_path`: core resolved to ATen path for a long sequence.
  - `cuda_core_head_dim_unsupported`: `head_dim>128`, so CUDA experimental core cannot be used.
- For fail-loud cache/streaming guard errors, reduce `CODEX_WAN_FUSED_V1_Q_CHUNK` and/or `CODEX_WAN_FUSED_V1_KV_CHUNK`.
