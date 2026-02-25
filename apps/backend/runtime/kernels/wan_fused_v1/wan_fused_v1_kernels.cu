// WAN fused attention V1 CUDA implementations.
//
// Note: v1.1 uses streaming tiled attention with online softmax accumulation to
// avoid materializing full LxL score/probability tensors in global VRAM.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <tuple>

namespace {

constexpr double kRmsNormEps = 1e-6;
constexpr int64_t kDefaultQChunk = 512;
constexpr int64_t kDefaultKvChunk = 1024;
constexpr int64_t kMaxQChunk = 512;
constexpr int64_t kMaxKvChunk = 1024;
constexpr int64_t kMaxScoreTileBytes = 128 * 1024 * 1024;
constexpr int64_t kMaxQKvTileAreaElements = 512 * 1024;
constexpr int64_t kSmallAttentionMatrixElementsBypass = 128 * 1024;

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), "", name, " must be a CUDA tensor");
}

void check_same_device(const torch::Tensor& reference, const torch::Tensor& value, const char* name) {
  TORCH_CHECK(
      reference.get_device() == value.get_device(),
      name,
      " must be on the same CUDA device as x (x=",
      reference.get_device(),
      " got=",
      value.get_device(),
      ")");
}

void check_optional_same_device(const torch::Tensor& reference, const c10::optional<torch::Tensor>& value, const char* name) {
  if (!value.has_value()) {
    return;
  }
  check_cuda_tensor(*value, name);
  check_same_device(reference, *value, name);
}

torch::Tensor rmsnorm_channels(const torch::Tensor& x_blc, const torch::Tensor& weight_c) {
  auto x_fp32 = x_blc.to(torch::kFloat);
  auto variance = x_fp32.pow(2).mean(-1, true);
  auto normalized = x_fp32 * torch::rsqrt(variance + kRmsNormEps);
  auto scaled = normalized * weight_c.to(torch::kFloat).view({1, 1, weight_c.size(0)});
  return scaled.to(x_blc.scalar_type());
}

torch::Tensor apply_rope_blhd(const torch::Tensor& x_blhd, const torch::Tensor& rope_cos, const torch::Tensor& rope_sin) {
  TORCH_CHECK(x_blhd.dim() == 4, "apply_rope_blhd expects x as [B,L,H,D]");
  TORCH_CHECK(rope_cos.dim() == 4 && rope_sin.dim() == 4, "apply_rope_blhd expects rope tensors as [1,L,1,D]");

  auto dtype = x_blhd.scalar_type();
  auto x_fp32 = x_blhd.to(torch::kFloat);

  auto x_even = x_fp32.slice(-1, 0, x_fp32.size(-1), 2);
  auto x_odd = x_fp32.slice(-1, 1, x_fp32.size(-1), 2);
  auto cos = rope_cos.slice(-1, 0, rope_cos.size(-1), 2).to(torch::kFloat);
  auto sin = rope_sin.slice(-1, 1, rope_sin.size(-1), 2).to(torch::kFloat);

  auto out_even = x_even * cos - x_odd * sin;
  auto out_odd = x_even * sin + x_odd * cos;

  auto out = torch::empty_like(x_fp32);
  out.slice(-1, 0, out.size(-1), 2).copy_(out_even);
  out.slice(-1, 1, out.size(-1), 2).copy_(out_odd);
  return out.to(dtype);
}

int64_t parse_env_chunk_or_default(const char* key, int64_t default_value, int64_t hard_cap) {
  const char* raw = std::getenv(key);
  if (raw == nullptr) {
    return default_value;
  }
  const std::string raw_value(raw);
  TORCH_CHECK(!raw_value.empty(), key, " must be a strict integer (got empty value)");
  for (char ch : raw_value) {
    TORCH_CHECK(ch >= '0' && ch <= '9', key, " must be a strict integer (got '", raw, "')");
  }
  std::size_t consumed = 0;
  long long parsed = 0;
  try {
    parsed = std::stoll(raw_value, &consumed, 10);
  } catch (...) {
    TORCH_CHECK(false, key, " must be a positive integer (got '", raw, "')");
  }
  TORCH_CHECK(consumed == raw_value.size(), key, " must be a strict integer (got '", raw, "')");
  TORCH_CHECK(parsed > 0, key, " must be > 0 (got ", parsed, ")");
  TORCH_CHECK(parsed <= hard_cap, key, " exceeds hard cap ", hard_cap, " for wan_fused_v1 v1.1");
  return static_cast<int64_t>(parsed);
}

int64_t checked_mul_int64(int64_t left, int64_t right, const char* label) {
  TORCH_CHECK(left >= 0 && right >= 0, label, " must use non-negative factors.");
  if (left == 0 || right == 0) {
    return 0;
  }
  TORCH_CHECK(left <= std::numeric_limits<int64_t>::max() / right, label, " overflow.");
  return left * right;
}

struct StreamingPlan {
  int64_t q_chunk;
  int64_t kv_chunk;
  int64_t full_attention_elements;
  int64_t score_tile_elements;
  int64_t score_tile_bytes;
};

StreamingPlan enforce_streaming_invariants(
    int64_t batch,
    int64_t heads,
    int64_t q_len,
    int64_t kv_len,
    int64_t q_chunk_size,
    int64_t kv_chunk_size) {
  int64_t resolved_q_chunk = std::max<int64_t>(1, std::min<int64_t>(q_len, q_chunk_size));
  int64_t resolved_kv_chunk = std::max<int64_t>(1, std::min<int64_t>(kv_len, kv_chunk_size));
  const int64_t attention_matrix_elements = checked_mul_int64(q_len, kv_len, "q_len*kv_len");
  if (attention_matrix_elements > kSmallAttentionMatrixElementsBypass &&
      resolved_q_chunk == q_len &&
      resolved_kv_chunk == kv_len) {
    if (kv_len > 1) {
      int64_t adaptive_kv_chunk =
          std::max<int64_t>(1, kSmallAttentionMatrixElementsBypass / std::max<int64_t>(1, q_len));
      adaptive_kv_chunk = std::min<int64_t>(adaptive_kv_chunk, kv_len - 1);
      if (adaptive_kv_chunk >= 1 && adaptive_kv_chunk < kv_len) {
        resolved_kv_chunk = adaptive_kv_chunk;
      }
    }
    if (resolved_q_chunk == q_len && resolved_kv_chunk == kv_len && q_len > 1) {
      int64_t adaptive_q_chunk =
          std::max<int64_t>(1, kSmallAttentionMatrixElementsBypass / std::max<int64_t>(1, kv_len));
      adaptive_q_chunk = std::min<int64_t>(adaptive_q_chunk, q_len - 1);
      if (adaptive_q_chunk >= 1 && adaptive_q_chunk < q_len) {
        resolved_q_chunk = adaptive_q_chunk;
      }
    }
  }

  const int64_t tile_area = checked_mul_int64(resolved_q_chunk, resolved_kv_chunk, "q_chunk*kv_chunk");
  TORCH_CHECK(
      tile_area <= kMaxQKvTileAreaElements,
      "streaming invariant violated: q_chunk*kv_chunk exceeds cap (",
      tile_area,
      " > ",
      kMaxQKvTileAreaElements,
      ").");

  const int64_t bh = checked_mul_int64(batch, heads, "batch*heads");
  const int64_t full_attention_elements =
      checked_mul_int64(checked_mul_int64(bh, q_len, "batch*heads*q_len"), kv_len, "full_attention_elements");
  const int64_t score_tile_elements =
      checked_mul_int64(checked_mul_int64(bh, resolved_q_chunk, "batch*heads*q_chunk"), resolved_kv_chunk, "score_tile_elements");
  const int64_t score_tile_bytes = checked_mul_int64(score_tile_elements, static_cast<int64_t>(sizeof(float)), "score_tile_bytes");

  TORCH_CHECK(
      score_tile_bytes <= kMaxScoreTileBytes,
      "streaming invariant violated: score tile bytes exceed cap (",
      score_tile_bytes,
      " > ",
      kMaxScoreTileBytes,
      "). Reduce ",
      "CODEX_WAN_FUSED_V1_Q_CHUNK or CODEX_WAN_FUSED_V1_KV_CHUNK.");

  if (attention_matrix_elements > kSmallAttentionMatrixElementsBypass) {
    TORCH_CHECK(
        resolved_q_chunk < q_len || resolved_kv_chunk < kv_len,
        "streaming invariant violated: long sequence would run as full attention tile (q_chunk==q_len and kv_chunk==kv_len).");
  }

  StreamingPlan plan;
  plan.q_chunk = resolved_q_chunk;
  plan.kv_chunk = resolved_kv_chunk;
  plan.full_attention_elements = full_attention_elements;
  plan.score_tile_elements = score_tile_elements;
  plan.score_tile_bytes = score_tile_bytes;
  return plan;
}

torch::Tensor streaming_attention_bhld(
    const torch::Tensor& q_bhld,
    const torch::Tensor& k_bhmd,
    const torch::Tensor& v_bhmd,
    int64_t q_chunk_size,
    int64_t kv_chunk_size) {
  TORCH_CHECK(q_bhld.dim() == 4 && k_bhmd.dim() == 4 && v_bhmd.dim() == 4, "attention tensors must be rank-4");
  TORCH_CHECK(q_bhld.size(0) == k_bhmd.size(0) && q_bhld.size(0) == v_bhmd.size(0), "batch mismatch");
  TORCH_CHECK(q_bhld.size(1) == k_bhmd.size(1) && q_bhld.size(1) == v_bhmd.size(1), "head mismatch");
  TORCH_CHECK(k_bhmd.size(2) == v_bhmd.size(2), "kv length mismatch");
  TORCH_CHECK(q_bhld.size(3) == k_bhmd.size(3) && q_bhld.size(3) == v_bhmd.size(3), "head_dim mismatch");

  const auto bsz = q_bhld.size(0);
  const auto heads = q_bhld.size(1);
  const auto q_len = q_bhld.size(2);
  const auto kv_len = k_bhmd.size(2);
  const auto head_dim = q_bhld.size(3);
  const auto device = q_bhld.device();
  const auto output_dtype = q_bhld.scalar_type();
  const auto options_fp32 = q_bhld.options().dtype(torch::kFloat);
  const auto options_out = q_bhld.options().dtype(output_dtype);
  const auto scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const StreamingPlan plan = enforce_streaming_invariants(
      bsz,
      heads,
      q_len,
      kv_len,
      q_chunk_size,
      kv_chunk_size);
  const int64_t resolved_q_chunk = plan.q_chunk;
  const int64_t resolved_kv_chunk = plan.kv_chunk;

  auto out_bhld = torch::empty({bsz, heads, q_len, head_dim}, options_out);

  for (int64_t q_start = 0; q_start < q_len; q_start += resolved_q_chunk) {
    const int64_t q_end = std::min<int64_t>(q_len, q_start + resolved_q_chunk);
    const int64_t q_span = q_end - q_start;

    auto q_chunk = q_bhld.slice(/*dim=*/2, q_start, q_end).to(torch::kFloat).contiguous();
    auto m = torch::full({bsz, heads, q_span, 1}, -std::numeric_limits<float>::infinity(), options_fp32.device(device));
    auto l = torch::zeros({bsz, heads, q_span, 1}, options_fp32.device(device));
    auto acc = torch::zeros({bsz, heads, q_span, head_dim}, options_fp32.device(device));

    for (int64_t kv_start = 0; kv_start < kv_len; kv_start += resolved_kv_chunk) {
      const int64_t kv_end = std::min<int64_t>(kv_len, kv_start + resolved_kv_chunk);
      auto k_chunk = k_bhmd.slice(/*dim=*/2, kv_start, kv_end).to(torch::kFloat).contiguous();
      auto v_chunk = v_bhmd.slice(/*dim=*/2, kv_start, kv_end).to(torch::kFloat).contiguous();

      auto scores = torch::matmul(q_chunk, k_chunk.transpose(-2, -1));
      scores.mul_(scale);
      auto max_chunk = std::get<0>(scores.max(/*dim=*/-1, /*keepdim=*/true));
      auto m_new = torch::maximum(m, max_chunk);
      auto alpha = torch::exp(m - m_new);
      scores.sub_(m_new);
      scores.exp_();
      auto l_new = alpha * l + scores.sum(/*dim=*/-1, /*keepdim=*/true);
      auto acc_new = alpha * acc + torch::matmul(scores, v_chunk);

      m = m_new;
      l = l_new;
      acc = acc_new;
    }

    auto out_chunk = (acc / l.clamp_min(1e-9f)).to(output_dtype);
    out_bhld.slice(/*dim=*/2, q_start, q_end).copy_(out_chunk);
  }

  return out_bhld;
}

torch::Tensor linear_lastdim(
    const torch::Tensor& x_blc,
    const torch::Tensor& w_ci,
    const c10::optional<torch::Tensor>& bias) {
  auto x2d = x_blc.contiguous().view({-1, x_blc.size(-1)});
  auto out2d = torch::matmul(x2d, w_ci);
  if (bias.has_value()) {
    out2d = out2d + bias->view({1, bias->size(0)});
  }
  auto out = out2d.view({x_blc.size(0), x_blc.size(1), out2d.size(-1)});
  return out;
}

torch::Tensor project_linear_chunk_bhld(
    const torch::Tensor& source_blc,
    int64_t start,
    int64_t end,
    const torch::Tensor& weight_ci,
    const c10::optional<torch::Tensor>& bias_i,
    int64_t batch,
    int64_t heads,
    int64_t head_dim) {
  TORCH_CHECK(start >= 0 && end >= start && end <= source_blc.size(1), "project_linear_chunk_bhld: invalid chunk range");
  const int64_t span = end - start;
  auto projected_blc = linear_lastdim(source_blc.slice(/*dim=*/1, start, end), weight_ci, bias_i);
  auto projected_blhd = projected_blc.contiguous().view({batch, span, heads, head_dim});
  return projected_blhd.permute({0, 2, 1, 3}).contiguous();
}

torch::Tensor project_norm_rope_chunk_bhld(
    const torch::Tensor& source_blc,
    int64_t start,
    int64_t end,
    const torch::Tensor& weight_ci,
    const c10::optional<torch::Tensor>& bias_i,
    const torch::Tensor& norm_weight,
    const torch::Tensor& rope_cos,
    const torch::Tensor& rope_sin,
    int64_t batch,
    int64_t heads,
    int64_t head_dim) {
  TORCH_CHECK(start >= 0 && end >= start && end <= source_blc.size(1), "project_norm_rope_chunk_bhld: invalid chunk range");
  const int64_t span = end - start;
  auto projected_blc = linear_lastdim(source_blc.slice(/*dim=*/1, start, end), weight_ci, bias_i);
  projected_blc = rmsnorm_channels(projected_blc, norm_weight);
  auto projected_blhd = projected_blc.contiguous().view({batch, span, heads, head_dim});
  auto rope_cos_chunk = rope_cos.slice(/*dim=*/1, start, end);
  auto rope_sin_chunk = rope_sin.slice(/*dim=*/1, start, end);
  projected_blhd = apply_rope_blhd(projected_blhd, rope_cos_chunk, rope_sin_chunk);
  return projected_blhd.permute({0, 2, 1, 3}).contiguous();
}

}  // namespace

torch::Tensor wan_fused_v1_self_fwd_cuda(
    const torch::Tensor& x,
    const torch::Tensor& w_q,
    const c10::optional<torch::Tensor>& b_q,
    const torch::Tensor& w_k,
    const c10::optional<torch::Tensor>& b_k,
    const torch::Tensor& w_v,
    const c10::optional<torch::Tensor>& b_v,
    const torch::Tensor& norm_q_weight,
    const torch::Tensor& norm_k_weight,
    const torch::Tensor& rope_cos_qk,
    const torch::Tensor& rope_sin_qk,
    const torch::Tensor& w_out,
    const c10::optional<torch::Tensor>& b_out) {
  check_cuda_tensor(x, "x");
  check_cuda_tensor(w_q, "w_q");
  check_cuda_tensor(w_k, "w_k");
  check_cuda_tensor(w_v, "w_v");
  check_cuda_tensor(norm_q_weight, "norm_q_weight");
  check_cuda_tensor(norm_k_weight, "norm_k_weight");
  check_cuda_tensor(rope_cos_qk, "rope_cos_qk");
  check_cuda_tensor(rope_sin_qk, "rope_sin_qk");
  check_cuda_tensor(w_out, "w_out");

  check_same_device(x, w_q, "w_q");
  check_same_device(x, w_k, "w_k");
  check_same_device(x, w_v, "w_v");
  check_same_device(x, norm_q_weight, "norm_q_weight");
  check_same_device(x, norm_k_weight, "norm_k_weight");
  check_same_device(x, rope_cos_qk, "rope_cos_qk");
  check_same_device(x, rope_sin_qk, "rope_sin_qk");
  check_same_device(x, w_out, "w_out");
  check_optional_same_device(x, b_q, "b_q");
  check_optional_same_device(x, b_k, "b_k");
  check_optional_same_device(x, b_v, "b_v");
  check_optional_same_device(x, b_out, "b_out");

  TORCH_CHECK(x.dim() == 3, "wan_fused_v1.self_fwd: x must be [B,L,C]");
  TORCH_CHECK(w_q.dim() == 3 && w_k.dim() == 3 && w_v.dim() == 3, "wan_fused_v1.self_fwd: w_q/w_k/w_v must be [C,H,D]");
  TORCH_CHECK(w_out.dim() == 3, "wan_fused_v1.self_fwd: w_out must be [H,D,C]");

  const auto bsz = x.size(0);
  const auto seq_len = x.size(1);
  const auto channels = x.size(2);
  TORCH_CHECK(w_q.size(0) == channels, "wan_fused_v1.self_fwd: w_q C mismatch");
  TORCH_CHECK(w_k.size(0) == channels, "wan_fused_v1.self_fwd: w_k C mismatch");
  TORCH_CHECK(w_v.size(0) == channels, "wan_fused_v1.self_fwd: w_v C mismatch");
  const auto num_heads = w_q.size(1);
  const auto head_dim = w_q.size(2);
  TORCH_CHECK(w_k.size(1) == num_heads && w_k.size(2) == head_dim, "wan_fused_v1.self_fwd: w_k H/D mismatch");
  TORCH_CHECK(w_v.size(1) == num_heads && w_v.size(2) == head_dim, "wan_fused_v1.self_fwd: w_v H/D mismatch");
  TORCH_CHECK(num_heads * head_dim == channels, "wan_fused_v1.self_fwd: H*D must equal C");
  TORCH_CHECK(w_out.size(0) == num_heads && w_out.size(1) == head_dim && w_out.size(2) == channels,
              "wan_fused_v1.self_fwd: w_out shape mismatch");
  TORCH_CHECK(
      rope_cos_qk.dim() == 4 && rope_cos_qk.size(0) == 1 && rope_cos_qk.size(1) == seq_len && rope_cos_qk.size(2) == 1 &&
          rope_cos_qk.size(3) == head_dim,
      "wan_fused_v1.self_fwd: rope_cos_qk must be [1,L,1,D]");
  TORCH_CHECK(
      rope_sin_qk.dim() == 4 && rope_sin_qk.size(0) == 1 && rope_sin_qk.size(1) == seq_len && rope_sin_qk.size(2) == 1 &&
          rope_sin_qk.size(3) == head_dim,
      "wan_fused_v1.self_fwd: rope_sin_qk must be [1,L,1,D]");
  TORCH_CHECK(norm_q_weight.numel() == channels && norm_k_weight.numel() == channels,
              "wan_fused_v1.self_fwd: norm weights must be [C]");

  const c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t q_chunk_size =
      parse_env_chunk_or_default("CODEX_WAN_FUSED_V1_Q_CHUNK", kDefaultQChunk, kMaxQChunk);
  const int64_t kv_chunk_size =
      parse_env_chunk_or_default("CODEX_WAN_FUSED_V1_KV_CHUNK", kDefaultKvChunk, kMaxKvChunk);

  auto wq = w_q.contiguous().view({channels, channels});
  auto wk = w_k.contiguous().view({channels, channels});
  auto wv = w_v.contiguous().view({channels, channels});

  const bool has_q_bias = b_q.has_value();
  const bool has_k_bias = b_k.has_value();
  const bool has_v_bias = b_v.has_value();
  TORCH_CHECK(
      has_q_bias == has_k_bias && has_q_bias == has_v_bias,
      "wan_fused_v1.self_fwd: requires all-or-none biases for q/k/v.");
  if (has_q_bias) {
    TORCH_CHECK(
        b_q->dim() == 2 && b_q->size(0) == num_heads && b_q->size(1) == head_dim,
        "wan_fused_v1.self_fwd: b_q must be [H,D]");
  }
  if (has_k_bias) {
    TORCH_CHECK(
        b_k->dim() == 2 && b_k->size(0) == num_heads && b_k->size(1) == head_dim,
        "wan_fused_v1.self_fwd: b_k must be [H,D]");
  }
  if (has_v_bias) {
    TORCH_CHECK(
        b_v->dim() == 2 && b_v->size(0) == num_heads && b_v->size(1) == head_dim,
        "wan_fused_v1.self_fwd: b_v must be [H,D]");
  }
  c10::optional<torch::Tensor> bq = has_q_bias ? c10::optional<torch::Tensor>(b_q->contiguous().view({channels})) : c10::nullopt;
  c10::optional<torch::Tensor> bk = has_k_bias ? c10::optional<torch::Tensor>(b_k->contiguous().view({channels})) : c10::nullopt;
  c10::optional<torch::Tensor> bv = has_v_bias ? c10::optional<torch::Tensor>(b_v->contiguous().view({channels})) : c10::nullopt;

  const StreamingPlan plan =
      enforce_streaming_invariants(bsz, num_heads, seq_len, seq_len, q_chunk_size, kv_chunk_size);
  const int64_t resolved_q_chunk = plan.q_chunk;
  const int64_t resolved_kv_chunk = plan.kv_chunk;

  auto k_cache = torch::empty({bsz, num_heads, seq_len, head_dim}, x.options());
  auto v_cache = torch::empty({bsz, num_heads, seq_len, head_dim}, x.options());
  for (int64_t kv_start = 0; kv_start < seq_len; kv_start += resolved_kv_chunk) {
    const int64_t kv_end = std::min<int64_t>(seq_len, kv_start + resolved_kv_chunk);
    auto k_chunk_bhld = project_norm_rope_chunk_bhld(
        x,
        kv_start,
        kv_end,
        wk,
        bk,
        norm_k_weight,
        rope_cos_qk,
        rope_sin_qk,
        bsz,
        num_heads,
        head_dim);
    auto v_chunk_bhld = project_linear_chunk_bhld(x, kv_start, kv_end, wv, bv, bsz, num_heads, head_dim);
    k_cache.slice(/*dim=*/2, kv_start, kv_end).copy_(k_chunk_bhld);
    v_cache.slice(/*dim=*/2, kv_start, kv_end).copy_(v_chunk_bhld);
  }

  auto out = torch::empty({bsz, seq_len, channels}, x.options());
  auto w_out_2d = w_out.contiguous().view({channels, channels});
  for (int64_t q_start = 0; q_start < seq_len; q_start += resolved_q_chunk) {
    const int64_t q_end = std::min<int64_t>(seq_len, q_start + resolved_q_chunk);
    const int64_t q_span = q_end - q_start;
    auto q_chunk_bhld = project_norm_rope_chunk_bhld(
        x,
        q_start,
        q_end,
        wq,
        bq,
        norm_q_weight,
        rope_cos_qk,
        rope_sin_qk,
        bsz,
        num_heads,
        head_dim);
    auto attn_chunk_bhld =
        streaming_attention_bhld(q_chunk_bhld, k_cache, v_cache, q_span, resolved_kv_chunk);
    auto attn_chunk_blc =
        attn_chunk_bhld.permute({0, 2, 1, 3}).contiguous().view({bsz, q_span, channels});
    auto out_chunk = linear_lastdim(attn_chunk_blc, w_out_2d, b_out);
    out.slice(/*dim=*/1, q_start, q_end).copy_(out_chunk);
  }
  return out;
}

torch::Tensor wan_fused_v1_cross_fwd_cuda(
    const torch::Tensor& x,
    const torch::Tensor& context,
    const torch::Tensor& w_q,
    const c10::optional<torch::Tensor>& b_q,
    const torch::Tensor& norm_q_weight,
    const torch::Tensor& rope_cos_q,
    const torch::Tensor& rope_sin_q,
    const torch::Tensor& w_k,
    const c10::optional<torch::Tensor>& b_k,
    const torch::Tensor& norm_k_weight,
    const torch::Tensor& rope_cos_k,
    const torch::Tensor& rope_sin_k,
    const torch::Tensor& w_v,
    const c10::optional<torch::Tensor>& b_v,
    const torch::Tensor& w_out,
    const c10::optional<torch::Tensor>& b_out) {
  check_cuda_tensor(x, "x");
  check_cuda_tensor(context, "context");
  check_cuda_tensor(w_q, "w_q");
  check_cuda_tensor(norm_q_weight, "norm_q_weight");
  check_cuda_tensor(rope_cos_q, "rope_cos_q");
  check_cuda_tensor(rope_sin_q, "rope_sin_q");
  check_cuda_tensor(w_k, "w_k");
  check_cuda_tensor(norm_k_weight, "norm_k_weight");
  check_cuda_tensor(rope_cos_k, "rope_cos_k");
  check_cuda_tensor(rope_sin_k, "rope_sin_k");
  check_cuda_tensor(w_v, "w_v");
  check_cuda_tensor(w_out, "w_out");

  check_same_device(x, context, "context");
  check_same_device(x, w_q, "w_q");
  check_same_device(x, norm_q_weight, "norm_q_weight");
  check_same_device(x, rope_cos_q, "rope_cos_q");
  check_same_device(x, rope_sin_q, "rope_sin_q");
  check_same_device(x, w_k, "w_k");
  check_same_device(x, norm_k_weight, "norm_k_weight");
  check_same_device(x, rope_cos_k, "rope_cos_k");
  check_same_device(x, rope_sin_k, "rope_sin_k");
  check_same_device(x, w_v, "w_v");
  check_same_device(x, w_out, "w_out");
  check_optional_same_device(x, b_q, "b_q");
  check_optional_same_device(x, b_k, "b_k");
  check_optional_same_device(x, b_v, "b_v");
  check_optional_same_device(x, b_out, "b_out");

  TORCH_CHECK(x.dim() == 3, "wan_fused_v1.cross_fwd: x must be [B,Lq,C]");
  TORCH_CHECK(context.dim() == 3, "wan_fused_v1.cross_fwd: context must be [B,Lk,Cctx]");
  TORCH_CHECK(x.size(0) == context.size(0), "wan_fused_v1.cross_fwd: x/context batch mismatch");

  const auto bsz = x.size(0);
  const auto q_len = x.size(1);
  const auto channels = x.size(2);
  const auto kv_len = context.size(1);
  const auto ctx_dim = context.size(2);

  TORCH_CHECK(w_q.dim() == 3 && w_q.size(0) == channels, "wan_fused_v1.cross_fwd: w_q must be [C,H,D]");
  TORCH_CHECK(w_k.dim() == 3 && w_k.size(0) == ctx_dim, "wan_fused_v1.cross_fwd: w_k must be [Cctx,H,D]");
  TORCH_CHECK(w_v.dim() == 3 && w_v.size(0) == ctx_dim, "wan_fused_v1.cross_fwd: w_v must be [Cctx,H,D]");

  const auto num_heads = w_q.size(1);
  const auto head_dim = w_q.size(2);

  TORCH_CHECK(w_k.size(1) == num_heads && w_k.size(2) == head_dim, "wan_fused_v1.cross_fwd: w_k H/D mismatch");
  TORCH_CHECK(w_v.size(1) == num_heads && w_v.size(2) == head_dim, "wan_fused_v1.cross_fwd: w_v H/D mismatch");
  TORCH_CHECK(num_heads * head_dim == channels, "wan_fused_v1.cross_fwd: H*D must equal C");

  TORCH_CHECK(w_out.dim() == 3 && w_out.size(0) == num_heads && w_out.size(1) == head_dim && w_out.size(2) == channels,
              "wan_fused_v1.cross_fwd: w_out must be [H,D,C]");

  TORCH_CHECK(norm_q_weight.numel() == channels && norm_k_weight.numel() == channels,
              "wan_fused_v1.cross_fwd: norm weights must be [C]");

  TORCH_CHECK(
      rope_cos_q.dim() == 4 && rope_cos_q.size(0) == 1 && rope_cos_q.size(1) == q_len && rope_cos_q.size(2) == 1 &&
          rope_cos_q.size(3) == head_dim,
              "wan_fused_v1.cross_fwd: rope_cos_q must be [1,Lq,1,D]");
  TORCH_CHECK(
      rope_sin_q.dim() == 4 && rope_sin_q.size(0) == 1 && rope_sin_q.size(1) == q_len && rope_sin_q.size(2) == 1 &&
          rope_sin_q.size(3) == head_dim,
              "wan_fused_v1.cross_fwd: rope_sin_q must be [1,Lq,1,D]");
  TORCH_CHECK(
      rope_cos_k.dim() == 4 && rope_cos_k.size(0) == 1 && rope_cos_k.size(1) == kv_len && rope_cos_k.size(2) == 1 &&
          rope_cos_k.size(3) == head_dim,
              "wan_fused_v1.cross_fwd: rope_cos_k must be [1,Lk,1,D]");
  TORCH_CHECK(
      rope_sin_k.dim() == 4 && rope_sin_k.size(0) == 1 && rope_sin_k.size(1) == kv_len && rope_sin_k.size(2) == 1 &&
          rope_sin_k.size(3) == head_dim,
              "wan_fused_v1.cross_fwd: rope_sin_k must be [1,Lk,1,D]");

  const c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t q_chunk_size =
      parse_env_chunk_or_default("CODEX_WAN_FUSED_V1_Q_CHUNK", kDefaultQChunk, kMaxQChunk);
  const int64_t kv_chunk_size =
      parse_env_chunk_or_default("CODEX_WAN_FUSED_V1_KV_CHUNK", kDefaultKvChunk, kMaxKvChunk);

  auto wq_2d = w_q.contiguous().view({channels, channels});
  auto wk_2d = w_k.contiguous().view({ctx_dim, channels});
  auto wv_2d = w_v.contiguous().view({ctx_dim, channels});

  c10::optional<torch::Tensor> bq_flat = c10::nullopt;
  c10::optional<torch::Tensor> bk_flat = c10::nullopt;
  c10::optional<torch::Tensor> bv_flat = c10::nullopt;
  if (b_q.has_value()) {
    TORCH_CHECK(b_q->dim() == 2 && b_q->size(0) == num_heads && b_q->size(1) == head_dim,
                "wan_fused_v1.cross_fwd: b_q must be [H,D]");
    bq_flat = b_q->contiguous().view({channels});
  }
  if (b_k.has_value()) {
    TORCH_CHECK(b_k->dim() == 2 && b_k->size(0) == num_heads && b_k->size(1) == head_dim,
                "wan_fused_v1.cross_fwd: b_k must be [H,D]");
    bk_flat = b_k->contiguous().view({channels});
  }
  if (b_v.has_value()) {
    TORCH_CHECK(b_v->dim() == 2 && b_v->size(0) == num_heads && b_v->size(1) == head_dim,
                "wan_fused_v1.cross_fwd: b_v must be [H,D]");
    bv_flat = b_v->contiguous().view({channels});
  }

  const StreamingPlan plan =
      enforce_streaming_invariants(bsz, num_heads, q_len, kv_len, q_chunk_size, kv_chunk_size);
  const int64_t resolved_q_chunk = plan.q_chunk;
  const int64_t resolved_kv_chunk = plan.kv_chunk;

  auto k_cache = torch::empty({bsz, num_heads, kv_len, head_dim}, x.options());
  auto v_cache = torch::empty({bsz, num_heads, kv_len, head_dim}, x.options());
  for (int64_t kv_start = 0; kv_start < kv_len; kv_start += resolved_kv_chunk) {
    const int64_t kv_end = std::min<int64_t>(kv_len, kv_start + resolved_kv_chunk);
    auto k_chunk_bhld = project_norm_rope_chunk_bhld(
        context,
        kv_start,
        kv_end,
        wk_2d,
        bk_flat,
        norm_k_weight,
        rope_cos_k,
        rope_sin_k,
        bsz,
        num_heads,
        head_dim);
    auto v_chunk_bhld =
        project_linear_chunk_bhld(context, kv_start, kv_end, wv_2d, bv_flat, bsz, num_heads, head_dim);
    k_cache.slice(/*dim=*/2, kv_start, kv_end).copy_(k_chunk_bhld);
    v_cache.slice(/*dim=*/2, kv_start, kv_end).copy_(v_chunk_bhld);
  }

  auto out = torch::empty({bsz, q_len, channels}, x.options());
  auto w_out_2d = w_out.contiguous().view({channels, channels});
  for (int64_t q_start = 0; q_start < q_len; q_start += resolved_q_chunk) {
    const int64_t q_end = std::min<int64_t>(q_len, q_start + resolved_q_chunk);
    const int64_t q_span = q_end - q_start;
    auto q_chunk_bhld = project_norm_rope_chunk_bhld(
        x,
        q_start,
        q_end,
        wq_2d,
        bq_flat,
        norm_q_weight,
        rope_cos_q,
        rope_sin_q,
        bsz,
        num_heads,
        head_dim);
    auto attn_chunk_bhld =
        streaming_attention_bhld(q_chunk_bhld, k_cache, v_cache, q_span, resolved_kv_chunk);
    auto attn_chunk_blc =
        attn_chunk_bhld.permute({0, 2, 1, 3}).contiguous().view({bsz, q_span, channels});
    auto out_chunk = linear_lastdim(attn_chunk_blc, w_out_2d, b_out);
    out.slice(/*dim=*/1, q_start, q_end).copy_(out_chunk);
  }
  return out;
}
