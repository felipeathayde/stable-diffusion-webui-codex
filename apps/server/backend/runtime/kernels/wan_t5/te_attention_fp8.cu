// Chunked attention on CUDA using ATen ops (no full LxL materialization).
// Q,K,V are expected in [B,H,L,D]. We process queries in tiles along L to limit memory.

#include <torch/extension.h>
using torch::Tensor;

namespace {

Tensor to_contig_cuda(const Tensor& t) {
  if (t.is_contiguous()) return t;
  return t.contiguous();
}

inline int64_t env_chunk_or(const char* name, int64_t defv) {
  const char* v = std::getenv(name);
  if (!v) return defv;
  try { return std::max<int64_t>(32, std::stoll(v)); } catch (...) { return defv; }
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
  int64_t chunk = env_chunk_or("WAN_TE_ATTN_CHUNK", 192);
  double scale = 1.0 / std::sqrt(static_cast<double>(D));

  const char* impl = std::getenv("WAN_TE_ATTN_IMPL");
  // Default to kernel when extension is present; allow override by setting WAN_TE_ATTN_IMPL=aten
  bool use_kernel = (!impl) || (std::string(impl) == std::string("kernel"));
  if (use_kernel) {
    // Call launcher that currently uses ATen chunked; swap to true kernel when ready
    extern std::vector<Tensor> attn_stream_rows_launcher(const Tensor&, const Tensor&, const Tensor&, const bool, const int64_t);
    auto ret = attn_stream_rows_launcher(q, k, v, causal, chunk);
    out.copy_(ret[0]);
  } else {
    for (int64_t s = 0; s < L; s += chunk) {
      int64_t c = std::min<int64_t>(chunk, L - s);
      Tensor q_chunk = qv.narrow(1, s, c);              // [B*H, c, D]
      Tensor logits = torch::matmul(q_chunk, kT) * scale; // [B*H, c, L]
      if (attn_mask.has_value()) {
        // Not wired yet; keep strict: disallow for now
        TORCH_CHECK(false, "attn_fp8_forward: attn_mask not supported in this version");
      }
      if (causal) {
        // Simple causal mask per chunk: disallow keys > current absolute position
        // Build a row-wise mask using arange on device
        Tensor arL = torch::arange(L, q.device());        // [L]
        for (int64_t i = 0; i < c; ++i) {
          int64_t pos = s + i;
          Tensor row = logits.select(1, i);               // [B*H, L]
          Tensor mask = arL.gt(pos).to(row.dtype()) * (-1e9);
          row.add_(mask);
        }
      }
      // Stable softmax: subtract max
      Tensor m = std::get<0>(logits.max(-1, true));
      Tensor p = (logits - m).softmax(-1);
      Tensor o_chunk = torch::matmul(p, vv);             // [B*H, c, D]
      out.reshape({B*H, L, D}).narrow(1, s, c).copy_(o_chunk);
    }
  }

  return {out, Tensor()};
}
