/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN T5 CUDA extension op implementing a memory-friendly attention forward path for FP8 TE runs.
Implements a chunked attention wrapper for Q/K/V in `[B,H,L,D]` and delegates to a streaming launcher when available.

Symbols (top-level; keep in sync; no ghosts):
- `to_contig_cuda` (function): Ensures a tensor is contiguous on CUDA before downstream ops.
- `te_attn_fp8_forward` (function): Chunked/streamed attention forward for the WAN TE FP8 path (returns output + optional probs).
*/

// Chunked attention on CUDA using ATen ops (no full LxL materialization).
// Q,K,V are expected in [B,H,L,D]. We process queries in tiles along L to limit memory.

#include <torch/extension.h>
using torch::Tensor;

namespace {

Tensor to_contig_cuda(const Tensor& t) {
  if (t.is_contiguous()) return t;
  return t.contiguous();
}

}

std::vector<Tensor> te_attn_fp8_forward(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const c10::optional<Tensor>& attn_mask, bool causal, const int8_t fp8_format) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "attn_fp8_forward: tensors must be CUDA");
  TORCH_CHECK(q.dim()==4 && k.dim()==4 && v.dim()==4, "attn_fp8_forward: expected [B,H,L,D]");
  // Assume activations in compute dtype (fp16/bf16). We only tile along queries.
  auto B = q.size(0); auto H = q.size(1); auto L = q.size(2); auto D = q.size(3);
  auto opts = q.options();
  Tensor out = torch::empty_like(q);

  // Reshape for batched matmul
  Tensor qv = q.reshape({B*H, L, D});
  Tensor kv = k.reshape({B*H, L, D});
  Tensor vv = v.reshape({B*H, L, D});
  Tensor kT = kv.transpose(1, 2); // [B*H, D, L]

  // Chunk size for queries
  int64_t chunk = 192;
  double scale = 1.0 / std::sqrt(static_cast<double>(D));

  // Call launcher that currently uses ATen chunked; swap to true kernel when ready
  extern std::vector<Tensor> attn_stream_rows_launcher(const Tensor&, const Tensor&, const Tensor&, const bool, const int64_t);
  auto ret = attn_stream_rows_launcher(q, k, v, causal, chunk);
  out.copy_(ret[0]);

  return {out, Tensor()};
}
