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
- 2026-02-25: Added explicit load-time warmup API (`warmup_extension_for_load`) that resolves fused mode/env gates, triggers extension load/build before denoise, and raises fail-loud when warmup fails under `force` mode.
- 2026-02-25: Kernel-runtime invariants are now mapped to explicit contract code `E_WAN_FUSED_STREAMING_INVARIANT_VIOLATION` so failures in streaming-only guarantees are surfaced without generic error masking.
- 2026-02-25: Self fused dispatch no longer packs `w_qkv`/`b_qkv`; wrapper now passes `w_q/w_k/w_v` + optional biases separately to cut transient VRAM overhead before kernel dispatch.
- 2026-02-25: Runtime loader enforces extension ABI (`WAN_FUSED_V1_ABI=2`) and rejects stale modules to avoid silent signature mismatch after self_fwd contract change.
- 2026-02-25: Loader now purges import cache for extension module names during stage fallback so an ABI-rejected prebuilt module does not poison in-place/JIT resolution in-process.
- 2026-02-26: Runtime now recognizes kernel-side telemetry controls (`CODEX_WAN_FUSED_V1_KERNEL_TRACE`, `CODEX_WAN_FUSED_V1_KERNEL_TRACE_KV`, `CODEX_WAN_FUSED_V1_KERNEL_TRACE_EVERY_Q`, `CODEX_WAN_FUSED_V1_KERNEL_TRACE_EVERY_KV`) for per-phase VRAM snapshots emitted by fused CUDA path.
- 2026-02-26: Added attention core selector env `CODEX_WAN_FUSED_V1_ATTN_CORE=aten|cuda_experimental` (default `aten`), allowing bootstrap of custom CUDA streaming attention-core updates while preserving ATen path fallback for unsupported tuples.
