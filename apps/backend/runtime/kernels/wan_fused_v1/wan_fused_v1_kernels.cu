// WAN fused attention V1 CUDA implementations.
//
// Note: v1.1 uses streaming tiled attention with online softmax accumulation to
// avoid materializing full LxL score/probability tensors in global VRAM.

#include <torch/extension.h>

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr double kRmsNormEps = 1e-6;
constexpr int64_t kDefaultQChunk = 512;
constexpr int64_t kDefaultKvChunk = 1024;
constexpr int64_t kMaxQChunk = 512;
constexpr int64_t kMaxKvChunk = 1024;
constexpr int64_t kMaxScoreTileBytes = 128 * 1024 * 1024;
constexpr int64_t kMaxKvPrecomputeWorkspaceBytes = 320 * 1024 * 1024;
constexpr int64_t kMaxQKvTileAreaElements = 512 * 1024;
constexpr int64_t kSmallAttentionMatrixElementsBypass = 128 * 1024;
constexpr int64_t kCudaCoreMaxHeadDim = 128;
constexpr const char* kAttentionCoreEnvKey = "CODEX_WAN_FUSED_V1_ATTN_CORE";
constexpr const char* kAttentionCoreValidValues = "aten | cuda_experimental";

enum class AttentionCoreMode {
  ATE_NAIVE = 0,
  CUDA_STREAMING_EXPERIMENTAL = 1,
};

static_assert(static_cast<int>(AttentionCoreMode::ATE_NAIVE) == 0, "AttentionCoreMode::ATE_NAIVE ABI drift");
static_assert(
    static_cast<int>(AttentionCoreMode::CUDA_STREAMING_EXPERIMENTAL) == 1,
    "AttentionCoreMode::CUDA_STREAMING_EXPERIMENTAL ABI drift");
static_assert(kCudaCoreMaxHeadDim <= 128, "CUDA core head_dim bound must remain Ampere-safe (<=128).");

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

bool parse_env_flag_or_default(const char* key, bool default_value) {
  const char* raw = std::getenv(key);
  if (raw == nullptr) {
    return default_value;
  }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (value == "1" || value == "true" || value == "on" || value == "yes") {
    return true;
  }
  if (value == "0" || value == "false" || value == "off" || value == "no") {
    return false;
  }
  TORCH_CHECK(
      false,
      key,
      " must be a boolean flag (accepted: 1/0,true/false,on/off,yes/no; got '",
      raw,
      "')");
  return default_value;
}

AttentionCoreMode parse_attention_core_mode_token(const std::string& raw_value, const char* source_label) {
  std::string value(raw_value);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (value == "aten" || value == "default" || value == "off") {
    return AttentionCoreMode::ATE_NAIVE;
  }
  if (value == "cuda" || value == "cuda_experimental") {
    return AttentionCoreMode::CUDA_STREAMING_EXPERIMENTAL;
  }
  TORCH_CHECK(
      false,
      source_label,
      " must be one of: ",
      kAttentionCoreValidValues,
      " (got '",
      raw_value,
      "')");
  return AttentionCoreMode::ATE_NAIVE;
}

AttentionCoreMode parse_attention_core_mode_from_env() {
  const char* raw = std::getenv(kAttentionCoreEnvKey);
  if (raw == nullptr) {
    return AttentionCoreMode::ATE_NAIVE;
  }
  return parse_attention_core_mode_token(std::string(raw), kAttentionCoreEnvKey);
}

AttentionCoreMode attention_core_mode(
    const c10::optional<std::string>& explicit_attn_core,
    const char* explicit_source_label) {
  if (explicit_attn_core.has_value() && !explicit_attn_core->empty()) {
    return parse_attention_core_mode_token(*explicit_attn_core, explicit_source_label);
  }
  return parse_attention_core_mode_from_env();
}

bool kernel_trace_enabled() {
  static const bool enabled = parse_env_flag_or_default("CODEX_WAN_FUSED_V1_KERNEL_TRACE", false);
  return enabled;
}

bool kernel_trace_kv_enabled() {
  static const bool enabled = parse_env_flag_or_default("CODEX_WAN_FUSED_V1_KERNEL_TRACE_KV", false);
  return enabled;
}

int64_t kernel_trace_every_q_chunk() {
  static const int64_t every =
      parse_env_chunk_or_default("CODEX_WAN_FUSED_V1_KERNEL_TRACE_EVERY_Q", 1, 4096);
  return every;
}

int64_t kernel_trace_every_kv_chunk() {
  static const int64_t every =
      parse_env_chunk_or_default("CODEX_WAN_FUSED_V1_KERNEL_TRACE_EVERY_KV", 1, 4096);
  return every;
}

bool should_trace_chunk(int64_t index, int64_t total, int64_t every) {
  if (total <= 0 || index < 0) {
    return false;
  }
  if (index == 0 || index == total - 1) {
    return true;
  }
  return every > 0 && (index % every) == 0;
}

int64_t bytes_to_mb_floor(int64_t value) {
  if (value < 0) {
    return -1;
  }
  return value / (1024 * 1024);
}

struct TraceMemorySnapshot {
  int64_t alloc_mb;
  int64_t reserved_mb;
  int64_t max_alloc_mb;
  int64_t max_reserved_mb;
  int64_t free_mb;
  int64_t total_mb;
};

TraceMemorySnapshot capture_trace_memory_snapshot(c10::DeviceIndex device_index) {
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);
  const auto aggregate_index = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  const auto alloc_bytes = stats.allocated_bytes[aggregate_index].current;
  const auto reserved_bytes = stats.reserved_bytes[aggregate_index].current;
  const auto max_alloc_bytes = stats.allocated_bytes[aggregate_index].peak;
  const auto max_reserved_bytes = stats.reserved_bytes[aggregate_index].peak;

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaError_t mem_status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (mem_status != cudaSuccess) {
    free_bytes = 0;
    total_bytes = 0;
  }

  TraceMemorySnapshot snapshot;
  snapshot.alloc_mb = bytes_to_mb_floor(alloc_bytes);
  snapshot.reserved_mb = bytes_to_mb_floor(reserved_bytes);
  snapshot.max_alloc_mb = bytes_to_mb_floor(max_alloc_bytes);
  snapshot.max_reserved_mb = bytes_to_mb_floor(max_reserved_bytes);
  snapshot.free_mb = mem_status == cudaSuccess ? bytes_to_mb_floor(static_cast<int64_t>(free_bytes)) : -1;
  snapshot.total_mb = mem_status == cudaSuccess ? bytes_to_mb_floor(static_cast<int64_t>(total_bytes)) : -1;
  return snapshot;
}

void maybe_trace_kernel_memory(
    const char* op_name,
    const char* phase_name,
    const torch::Tensor& tensor_anchor,
    int64_t q_start,
    int64_t q_end,
    int64_t q_len,
    int64_t kv_start,
    int64_t kv_end,
    int64_t kv_len) {
  if (!kernel_trace_enabled()) {
    return;
  }
  const auto device_index = tensor_anchor.get_device();
  const auto snapshot = capture_trace_memory_snapshot(device_index);
  const char* dtype_name = c10::toString(tensor_anchor.scalar_type());
  std::fprintf(
      stderr,
      "[wan_fused_v1.trace] op=%s phase=%s dtype=%s device=cuda:%d q=%lld:%lld/%lld kv=%lld:%lld/%lld "
      "alloc=%lldMB reserved=%lldMB free=%lldMB total=%lldMB max_alloc=%lldMB max_reserved=%lldMB\n",
      op_name,
      phase_name,
      dtype_name,
      static_cast<int>(device_index),
      static_cast<long long>(q_start),
      static_cast<long long>(q_end),
      static_cast<long long>(q_len),
      static_cast<long long>(kv_start),
      static_cast<long long>(kv_end),
      static_cast<long long>(kv_len),
      static_cast<long long>(snapshot.alloc_mb),
      static_cast<long long>(snapshot.reserved_mb),
      static_cast<long long>(snapshot.free_mb),
      static_cast<long long>(snapshot.total_mb),
      static_cast<long long>(snapshot.max_alloc_mb),
      static_cast<long long>(snapshot.max_reserved_mb));
  std::fflush(stderr);
}

__global__ void streaming_attention_update_fp32_kernel(
    const float* q_ptr,
    const float* k_ptr,
    const float* v_ptr,
    float* m_ptr,
    float* l_ptr,
    float* acc_ptr,
    int64_t bh,
    int64_t q_span,
    int64_t kv_span,
    int64_t head_dim,
    float scale) {
  const int64_t row_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_rows = bh * q_span;
  if (row_index >= total_rows) {
    return;
  }

  const int64_t bh_index = row_index / q_span;
  const int64_t q_index = row_index - (bh_index * q_span);
  const int64_t q_offset = ((bh_index * q_span) + q_index) * head_dim;
  const int64_t kv_offset = bh_index * kv_span * head_dim;

  const float* q_row = q_ptr + q_offset;
  const float* k_base = k_ptr + kv_offset;
  const float* v_base = v_ptr + kv_offset;

  float local_max = -std::numeric_limits<float>::infinity();
  for (int64_t kv_index = 0; kv_index < kv_span; ++kv_index) {
    const float* k_row = k_base + kv_index * head_dim;
    float dot = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      dot += q_row[d] * k_row[d];
    }
    dot *= scale;
    local_max = fmaxf(local_max, dot);
  }

  float local_weight_sum = 0.0f;
  float local_weighted_value[kCudaCoreMaxHeadDim];
  for (int64_t d = 0; d < kCudaCoreMaxHeadDim; ++d) {
    local_weighted_value[d] = 0.0f;
  }
  for (int64_t kv_index = 0; kv_index < kv_span; ++kv_index) {
    const float* k_row = k_base + kv_index * head_dim;
    const float* v_row = v_base + kv_index * head_dim;
    float dot = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      dot += q_row[d] * k_row[d];
    }
    dot *= scale;
    const float weight = expf(dot - local_max);
    local_weight_sum += weight;
    for (int64_t d = 0; d < head_dim; ++d) {
      local_weighted_value[d] += weight * v_row[d];
    }
  }

  float* m_row = m_ptr + row_index;
  float* l_row = l_ptr + row_index;
  float* acc_row = acc_ptr + q_offset;

  const float old_m = *m_row;
  const float old_l = *l_row;
  const float new_m = fmaxf(old_m, local_max);
  const float alpha = expf(old_m - new_m);
  const float beta = expf(local_max - new_m);
  const float new_l = alpha * old_l + beta * local_weight_sum;

  for (int64_t d = 0; d < head_dim; ++d) {
    acc_row[d] = alpha * acc_row[d] + beta * local_weighted_value[d];
  }
  *m_row = new_m;
  *l_row = new_l;
}

void launch_streaming_attention_update_fp32(
    const torch::Tensor& q_fp32,
    const torch::Tensor& k_fp32,
    const torch::Tensor& v_fp32,
    torch::Tensor& m_fp32,
    torch::Tensor& l_fp32,
    torch::Tensor& acc_fp32,
    int64_t head_dim) {
  TORCH_CHECK(q_fp32.is_cuda() && k_fp32.is_cuda() && v_fp32.is_cuda(), "CUDA core expects CUDA tensors");
  TORCH_CHECK(q_fp32.scalar_type() == torch::kFloat, "CUDA core expects q as float32");
  TORCH_CHECK(k_fp32.scalar_type() == torch::kFloat, "CUDA core expects k as float32");
  TORCH_CHECK(v_fp32.scalar_type() == torch::kFloat, "CUDA core expects v as float32");
  TORCH_CHECK(m_fp32.scalar_type() == torch::kFloat, "CUDA core expects m as float32");
  TORCH_CHECK(l_fp32.scalar_type() == torch::kFloat, "CUDA core expects l as float32");
  TORCH_CHECK(acc_fp32.scalar_type() == torch::kFloat, "CUDA core expects acc as float32");
  TORCH_CHECK(q_fp32.is_contiguous() && k_fp32.is_contiguous() && v_fp32.is_contiguous(), "CUDA core expects contiguous q/k/v");
  TORCH_CHECK(m_fp32.is_contiguous() && l_fp32.is_contiguous() && acc_fp32.is_contiguous(), "CUDA core expects contiguous m/l/acc");
  TORCH_CHECK(head_dim > 0 && head_dim <= kCudaCoreMaxHeadDim, "CUDA core head_dim must be in [1,128]");

  const int64_t batch = q_fp32.size(0);
  const int64_t heads = q_fp32.size(1);
  const int64_t q_span = q_fp32.size(2);
  const int64_t kv_span = k_fp32.size(2);
  const int64_t bh = batch * heads;
  TORCH_CHECK(k_fp32.size(0) == batch && k_fp32.size(1) == heads, "CUDA core k shape mismatch");
  TORCH_CHECK(v_fp32.size(0) == batch && v_fp32.size(1) == heads, "CUDA core v shape mismatch");
  TORCH_CHECK(v_fp32.size(2) == kv_span, "CUDA core v kv mismatch");
  TORCH_CHECK(k_fp32.size(3) == head_dim && v_fp32.size(3) == head_dim, "CUDA core head_dim mismatch");
  TORCH_CHECK(m_fp32.numel() == bh * q_span, "CUDA core m shape mismatch");
  TORCH_CHECK(l_fp32.numel() == bh * q_span, "CUDA core l shape mismatch");
  TORCH_CHECK(acc_fp32.numel() == bh * q_span * head_dim, "CUDA core acc shape mismatch");

  const int threads = 128;
  const int64_t total_rows = bh * q_span;
  const int blocks = static_cast<int>((total_rows + threads - 1) / threads);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q_fp32.get_device());
  streaming_attention_update_fp32_kernel<<<blocks, threads, 0, stream>>>(
      q_fp32.data_ptr<float>(),
      k_fp32.data_ptr<float>(),
      v_fp32.data_ptr<float>(),
      m_fp32.data_ptr<float>(),
      l_fp32.data_ptr<float>(),
      acc_fp32.data_ptr<float>(),
      bh,
      q_span,
      kv_span,
      head_dim,
      scale);
  const cudaError_t launch_status = cudaGetLastError();
  TORCH_CHECK(
      launch_status == cudaSuccess,
      "CUDA core launch failed: ",
      cudaGetErrorString(launch_status));
}

int64_t checked_mul_int64(int64_t left, int64_t right, const char* label) {
  TORCH_CHECK(left >= 0 && right >= 0, label, " must use non-negative factors.");
  if (left == 0 || right == 0) {
    return 0;
  }
  TORCH_CHECK(left <= std::numeric_limits<int64_t>::max() / right, label, " overflow.");
  return left * right;
}

int64_t checked_add_int64(int64_t left, int64_t right, const char* label) {
  TORCH_CHECK(left >= 0 && right >= 0, label, " must use non-negative factors.");
  TORCH_CHECK(left <= std::numeric_limits<int64_t>::max() - right, label, " overflow.");
  return left + right;
}

void maybe_warn_streaming_chunk_downshift_once(
    int64_t requested_q_chunk,
    int64_t requested_kv_chunk,
    int64_t chosen_q_chunk,
    int64_t chosen_kv_chunk,
    int64_t tile_area_budget,
    int64_t tile_area_cap_by_score_bytes,
    int64_t tile_area_cap_by_workspace_bytes,
    int64_t remaining_bytes_for_score_tile,
    int64_t kv_cache_bytes) {
  if (requested_q_chunk == chosen_q_chunk && requested_kv_chunk == chosen_kv_chunk) {
    return;
  }
  static std::once_flag warning_once_flag;
  std::call_once(warning_once_flag, [&]() {
    std::fprintf(
        stderr,
        "[wan_fused_v1.warn] streaming chunk auto-downshift requested(q=%lld,kv=%lld) -> chosen(q=%lld,kv=%lld) "
        "tile_area_budget=%lld cap_area=%lld cap_score_bytes=%lld cap_workspace=%lld remaining_score_tile_bytes=%lld kv_cache_bytes=%lld\n",
        static_cast<long long>(requested_q_chunk),
        static_cast<long long>(requested_kv_chunk),
        static_cast<long long>(chosen_q_chunk),
        static_cast<long long>(chosen_kv_chunk),
        static_cast<long long>(tile_area_budget),
        static_cast<long long>(kMaxQKvTileAreaElements),
        static_cast<long long>(tile_area_cap_by_score_bytes),
        static_cast<long long>(tile_area_cap_by_workspace_bytes),
        static_cast<long long>(remaining_bytes_for_score_tile),
        static_cast<long long>(kv_cache_bytes));
    std::fflush(stderr);
  });
}

struct StreamingPlan {
  int64_t q_chunk;
  int64_t kv_chunk;
  int64_t full_attention_elements;
  int64_t score_tile_elements;
  int64_t score_tile_bytes;
  int64_t kv_cache_elements;
  int64_t kv_cache_bytes;
  int64_t precompute_workspace_bytes;
};

torch::Tensor project_linear_chunk_bhld(
    const torch::Tensor& source_blc,
    int64_t start,
    int64_t end,
    const torch::Tensor& weight_ci,
    const c10::optional<torch::Tensor>& bias_i,
    int64_t batch,
    int64_t heads,
    int64_t head_dim);

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
    int64_t head_dim);

StreamingPlan enforce_streaming_invariants(
    int64_t batch,
    int64_t heads,
    int64_t q_len,
    int64_t kv_len,
    int64_t head_dim,
    int64_t kv_cache_element_bytes,
    int64_t q_chunk_size,
    int64_t kv_chunk_size) {
  TORCH_CHECK(head_dim > 0, "streaming invariant violated: head_dim must be > 0.");
  TORCH_CHECK(
      kv_cache_element_bytes > 0,
      "streaming invariant violated: kv cache element bytes must be > 0 (got ",
      kv_cache_element_bytes,
      ").");
  const int64_t bh = checked_mul_int64(batch, heads, "batch*heads");
  const int64_t kv_cache_elements_per_tensor =
      checked_mul_int64(checked_mul_int64(bh, kv_len, "batch*heads*kv_len"), head_dim, "kv_cache_elements_per_tensor");
  const int64_t kv_cache_elements = checked_mul_int64(kv_cache_elements_per_tensor, 2, "kv_cache_elements");
  const int64_t kv_cache_bytes = checked_mul_int64(kv_cache_elements, kv_cache_element_bytes, "kv_cache_bytes");
  const int64_t requested_q_chunk = std::max<int64_t>(1, std::min<int64_t>(q_len, q_chunk_size));
  const int64_t requested_kv_chunk = std::max<int64_t>(1, std::min<int64_t>(kv_len, kv_chunk_size));
  const int64_t attention_matrix_elements = checked_mul_int64(q_len, kv_len, "q_len*kv_len");

  TORCH_CHECK(
      bh > 0,
      "streaming invariant violated: batch*heads must be > 0 (got ",
      bh,
      ").");
  const int64_t score_tile_bytes_per_area_element =
      checked_mul_int64(bh, static_cast<int64_t>(sizeof(float)), "score_tile_bytes_per_area_element");
  const int64_t tile_area_cap_by_score_bytes = kMaxScoreTileBytes / score_tile_bytes_per_area_element;

  const int64_t remaining_bytes_for_score_tile = kMaxKvPrecomputeWorkspaceBytes - kv_cache_bytes;
  TORCH_CHECK(
      remaining_bytes_for_score_tile > 0,
      "streaming invariant violated: full K/V precompute workspace cap exhausted by kv cache alone. "
      "bh=",
      bh,
      ", head_dim=",
      head_dim,
      ", q_len=",
      q_len,
      ", kv_len=",
      kv_len,
      ", kv_cache_bytes=",
      kv_cache_bytes,
      ", workspace_cap_bytes=",
      kMaxKvPrecomputeWorkspaceBytes,
      ", requested_q_chunk_size=",
      q_chunk_size,
      ", requested_kv_chunk_size=",
      kv_chunk_size,
      ".");
  const int64_t tile_area_cap_by_workspace_bytes =
      remaining_bytes_for_score_tile / score_tile_bytes_per_area_element;
  const int64_t tile_area_budget =
      std::min<int64_t>(kMaxQKvTileAreaElements, std::min<int64_t>(tile_area_cap_by_score_bytes, tile_area_cap_by_workspace_bytes));

  TORCH_CHECK(
      tile_area_budget > 0,
      "streaming invariant violated: no score-tile area budget remains after caps. "
      "bh=",
      bh,
      ", head_dim=",
      head_dim,
      ", q_len=",
      q_len,
      ", kv_len=",
      kv_len,
      ", kv_cache_bytes=",
      kv_cache_bytes,
      ", workspace_cap_bytes=",
      kMaxKvPrecomputeWorkspaceBytes,
      ", remaining_bytes_for_score_tile=",
      remaining_bytes_for_score_tile,
      ", tile_area_budget=",
      tile_area_budget,
      ", requested_q_chunk_size=",
      q_chunk_size,
      ", requested_kv_chunk_size=",
      kv_chunk_size,
      ". require q_chunk*kv_chunk <= ",
      tile_area_budget,
      ".");

  static constexpr int64_t kQChunkCandidatesDesc[] = {512, 256, 128, 64, 32, 16, 8, 4, 2, 1};
  static constexpr int64_t kKvChunkCandidatesDesc[] = {1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1};
  std::vector<int64_t> q_candidates;
  std::vector<int64_t> kv_candidates;
  q_candidates.reserve(sizeof(kQChunkCandidatesDesc) / sizeof(kQChunkCandidatesDesc[0]));
  kv_candidates.reserve(sizeof(kKvChunkCandidatesDesc) / sizeof(kKvChunkCandidatesDesc[0]));
  for (const int64_t candidate_q : kQChunkCandidatesDesc) {
    if (candidate_q <= requested_q_chunk && candidate_q <= q_len) {
      q_candidates.push_back(candidate_q);
    }
  }
  for (const int64_t candidate_kv : kKvChunkCandidatesDesc) {
    if (candidate_kv <= requested_kv_chunk && candidate_kv <= kv_len) {
      kv_candidates.push_back(candidate_kv);
    }
  }
  TORCH_CHECK(
      !q_candidates.empty() && !kv_candidates.empty(),
      "streaming invariant violated: no valid power-of-two chunk candidates under requested limits. "
      "bh=",
      bh,
      ", head_dim=",
      head_dim,
      ", q_len=",
      q_len,
      ", kv_len=",
      kv_len,
      ", requested_q_chunk_size=",
      q_chunk_size,
      ", requested_kv_chunk_size=",
      kv_chunk_size,
      ", requested_clamped_q_chunk=",
      requested_q_chunk,
      ", requested_clamped_kv_chunk=",
      requested_kv_chunk,
      ".");

  bool have_choice = false;
  int64_t resolved_q_chunk = 0;
  int64_t resolved_kv_chunk = 0;
  int64_t best_tile_area = 0;
  int64_t best_ratio_numerator = 0;
  int64_t best_ratio_denominator = 1;
  int64_t best_min_side = 0;
  const bool forbid_full_tile =
      attention_matrix_elements > kSmallAttentionMatrixElementsBypass;
  for (const int64_t candidate_q : q_candidates) {
    for (const int64_t candidate_kv : kv_candidates) {
      if (forbid_full_tile && candidate_q == q_len && candidate_kv == kv_len) {
        continue;
      }
      const int64_t candidate_tile_area =
          checked_mul_int64(candidate_q, candidate_kv, "candidate_q*candidate_kv");
      if (candidate_tile_area > tile_area_budget) {
        continue;
      }
      const int64_t candidate_min_side = std::min<int64_t>(candidate_q, candidate_kv);
      const int64_t candidate_max_side = std::max<int64_t>(candidate_q, candidate_kv);
      const int64_t candidate_ratio_numerator = candidate_max_side;
      const int64_t candidate_ratio_denominator = candidate_min_side;

      bool choose_candidate = false;
      if (!have_choice) {
        choose_candidate = true;
      } else if (candidate_tile_area > best_tile_area) {
        choose_candidate = true;
      } else if (candidate_tile_area == best_tile_area) {
        const int64_t ratio_left = checked_mul_int64(
            candidate_ratio_numerator,
            best_ratio_denominator,
            "candidate_ratio_numerator*best_ratio_denominator");
        const int64_t ratio_right = checked_mul_int64(
            best_ratio_numerator,
            candidate_ratio_denominator,
            "best_ratio_numerator*candidate_ratio_denominator");
        if (ratio_left < ratio_right) {
          choose_candidate = true;
        } else if (ratio_left == ratio_right) {
          if (candidate_min_side > best_min_side) {
            choose_candidate = true;
          } else if (candidate_min_side == best_min_side && candidate_q > resolved_q_chunk) {
            choose_candidate = true;
          }
        }
      }

      if (choose_candidate) {
        have_choice = true;
        resolved_q_chunk = candidate_q;
        resolved_kv_chunk = candidate_kv;
        best_tile_area = candidate_tile_area;
        best_ratio_numerator = candidate_ratio_numerator;
        best_ratio_denominator = candidate_ratio_denominator;
        best_min_side = candidate_min_side;
      }
    }
  }

  TORCH_CHECK(
      have_choice,
      "streaming invariant violated: no feasible (q_chunk,kv_chunk) pair fits computed budgets. "
      "bh=",
      bh,
      ", head_dim=",
      head_dim,
      ", q_len=",
      q_len,
      ", kv_len=",
      kv_len,
      ", kv_cache_bytes=",
      kv_cache_bytes,
      ", workspace_cap_bytes=",
      kMaxKvPrecomputeWorkspaceBytes,
      ", remaining_bytes_for_score_tile=",
      remaining_bytes_for_score_tile,
      ", tile_area_budget=",
      tile_area_budget,
      ", requested_q_chunk_size=",
      q_chunk_size,
      ", requested_kv_chunk_size=",
      kv_chunk_size,
      ". require q_chunk*kv_chunk <= ",
      tile_area_budget,
      ".");

  maybe_warn_streaming_chunk_downshift_once(
      requested_q_chunk,
      requested_kv_chunk,
      resolved_q_chunk,
      resolved_kv_chunk,
      tile_area_budget,
      tile_area_cap_by_score_bytes,
      tile_area_cap_by_workspace_bytes,
      remaining_bytes_for_score_tile,
      kv_cache_bytes);

  const int64_t tile_area =
      checked_mul_int64(resolved_q_chunk, resolved_kv_chunk, "q_chunk*kv_chunk");
  TORCH_CHECK(
      tile_area <= kMaxQKvTileAreaElements,
      "streaming invariant violated: q_chunk*kv_chunk exceeds cap (",
      tile_area,
      " > ",
      kMaxQKvTileAreaElements,
      ").");
  TORCH_CHECK(
      tile_area <= tile_area_budget,
      "streaming invariant violated: q_chunk*kv_chunk exceeds computed budget (",
      tile_area,
      " > ",
      tile_area_budget,
      ").");

  const int64_t full_attention_elements =
      checked_mul_int64(
          checked_mul_int64(bh, q_len, "batch*heads*q_len"),
          kv_len,
          "full_attention_elements");
  const int64_t score_tile_elements = checked_mul_int64(
      checked_mul_int64(bh, resolved_q_chunk, "batch*heads*q_chunk"),
      resolved_kv_chunk,
      "score_tile_elements");
  const int64_t score_tile_bytes = checked_mul_int64(
      score_tile_elements,
      static_cast<int64_t>(sizeof(float)),
      "score_tile_bytes");
  const int64_t precompute_workspace_bytes =
      checked_add_int64(score_tile_bytes, kv_cache_bytes, "precompute_workspace_bytes");

  TORCH_CHECK(
      score_tile_bytes <= kMaxScoreTileBytes,
      "streaming invariant violated: score tile bytes exceed cap (",
      score_tile_bytes,
      " > ",
      kMaxScoreTileBytes,
      "). bh=",
      bh,
      ", head_dim=",
      head_dim,
      ", q_len=",
      q_len,
      ", kv_len=",
      kv_len,
      ", kv_cache_bytes=",
      kv_cache_bytes,
      ", workspace_cap_bytes=",
      kMaxKvPrecomputeWorkspaceBytes,
      ", remaining_bytes_for_score_tile=",
      remaining_bytes_for_score_tile,
      ", tile_area_budget=",
      tile_area_budget,
      ", requested_q_chunk_size=",
      q_chunk_size,
      ", requested_kv_chunk_size=",
      kv_chunk_size,
      ". require q_chunk*kv_chunk <= ",
      tile_area_budget,
      ".");

  // F2/F7: full K/V precompute persists across the entire Q loop; guard this
  // persistent residency plus score tile workspace, not score tile alone.
  TORCH_CHECK(
      precompute_workspace_bytes <= kMaxKvPrecomputeWorkspaceBytes,
      "streaming invariant violated: full K/V precompute workspace bytes exceed cap (",
      precompute_workspace_bytes,
      " > ",
      kMaxKvPrecomputeWorkspaceBytes,
      "). bh=",
      bh,
      ", head_dim=",
      head_dim,
      ", q_len=",
      q_len,
      ", kv_len=",
      kv_len,
      ", kv_cache_bytes=",
      kv_cache_bytes,
      ", workspace_cap_bytes=",
      kMaxKvPrecomputeWorkspaceBytes,
      ", remaining_bytes_for_score_tile=",
      remaining_bytes_for_score_tile,
      ", tile_area_budget=",
      tile_area_budget,
      ", requested_q_chunk_size=",
      q_chunk_size,
      ", requested_kv_chunk_size=",
      kv_chunk_size,
      ". require q_chunk*kv_chunk <= ",
      tile_area_budget,
      ".");

  if (forbid_full_tile) {
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
  plan.kv_cache_elements = kv_cache_elements;
  plan.kv_cache_bytes = kv_cache_bytes;
  plan.precompute_workspace_bytes = precompute_workspace_bytes;
  return plan;
}

torch::Tensor streaming_attention_self_chunk_bhld(
    const torch::Tensor& q_chunk_bhld,
    const torch::Tensor& k_cache_bhld,
    const torch::Tensor& v_cache_bhld,
    int64_t kv_chunk_size,
    int64_t batch,
    int64_t heads,
    int64_t head_dim,
    AttentionCoreMode core_mode,
    bool trace_q_chunk,
    int64_t q_start,
    int64_t q_end,
    int64_t q_len_total) {
  TORCH_CHECK(q_chunk_bhld.dim() == 4, "streaming_attention_self_chunk_bhld: q_chunk must be [B,H,Lq,D]");
  TORCH_CHECK(k_cache_bhld.dim() == 4, "streaming_attention_self_chunk_bhld: k_cache must be [B,H,Lk,D]");
  TORCH_CHECK(v_cache_bhld.dim() == 4, "streaming_attention_self_chunk_bhld: v_cache must be [B,H,Lk,D]");
  TORCH_CHECK(
      k_cache_bhld.size(0) == batch && k_cache_bhld.size(1) == heads && k_cache_bhld.size(3) == head_dim,
      "streaming_attention_self_chunk_bhld: k_cache shape mismatch");
  TORCH_CHECK(
      v_cache_bhld.size(0) == batch && v_cache_bhld.size(1) == heads && v_cache_bhld.size(3) == head_dim,
      "streaming_attention_self_chunk_bhld: v_cache shape mismatch");
  TORCH_CHECK(
      k_cache_bhld.size(2) == v_cache_bhld.size(2),
      "streaming_attention_self_chunk_bhld: k_cache/v_cache sequence mismatch");
  const int64_t q_span = q_chunk_bhld.size(2);
  const int64_t kv_len = k_cache_bhld.size(2);
  const auto device = q_chunk_bhld.device();
  const auto output_dtype = q_chunk_bhld.scalar_type();
  const auto options_fp32 = q_chunk_bhld.options().dtype(torch::kFloat);
  const bool use_cuda_core = core_mode == AttentionCoreMode::CUDA_STREAMING_EXPERIMENTAL && head_dim <= kCudaCoreMaxHeadDim;
  const int64_t attention_elements = checked_mul_int64(q_len_total, kv_len, "self_attention_elements");
  if (trace_q_chunk && core_mode == AttentionCoreMode::ATE_NAIVE && attention_elements > kSmallAttentionMatrixElementsBypass) {
    maybe_trace_kernel_memory("self_attn", "aten_long_seq_path", q_chunk_bhld, q_start, q_end, q_len_total, 0, kv_len, kv_len);
  }
  if (trace_q_chunk && core_mode == AttentionCoreMode::CUDA_STREAMING_EXPERIMENTAL && !use_cuda_core) {
    maybe_trace_kernel_memory("self_attn", "cuda_core_head_dim_unsupported", q_chunk_bhld, q_start, q_end, q_len_total, 0, kv_len, kv_len);
  }
  auto q_chunk = q_chunk_bhld.to(torch::kFloat).contiguous();
  if (trace_q_chunk) {
    maybe_trace_kernel_memory("self_attn", "attn_chunk.q_fp32", q_chunk, q_start, q_end, q_len_total, 0, 0, kv_len);
  }
  auto m = torch::full({batch, heads, q_span, 1}, -std::numeric_limits<float>::infinity(), options_fp32.device(device));
  auto l = torch::zeros({batch, heads, q_span, 1}, options_fp32.device(device));
  auto acc = torch::zeros({batch, heads, q_span, head_dim}, options_fp32.device(device));

  const int64_t kv_span_full = std::min<int64_t>(kv_len, kv_chunk_size);
  auto k_chunk_full = torch::empty({batch, heads, kv_span_full, head_dim}, options_fp32.device(device));
  auto v_chunk_full = torch::empty({batch, heads, kv_span_full, head_dim}, options_fp32.device(device));
  const int64_t kv_span_tail = (kv_len > kv_chunk_size) ? (kv_len % kv_chunk_size) : 0;
  c10::optional<torch::Tensor> k_chunk_tail = c10::nullopt;
  c10::optional<torch::Tensor> v_chunk_tail = c10::nullopt;
  if (kv_span_tail != 0) {
    k_chunk_tail = torch::empty({batch, heads, kv_span_tail, head_dim}, options_fp32.device(device));
    v_chunk_tail = torch::empty({batch, heads, kv_span_tail, head_dim}, options_fp32.device(device));
  }

  const int64_t kv_total_chunks = (kv_len + kv_chunk_size - 1) / kv_chunk_size;
  const int64_t kv_trace_every = kernel_trace_every_kv_chunk();
  for (int64_t kv_start = 0; kv_start < kv_len; kv_start += kv_chunk_size) {
    const int64_t kv_end = std::min<int64_t>(kv_len, kv_start + kv_chunk_size);
    const int64_t kv_index = kv_start / kv_chunk_size;
    const bool trace_kv_chunk =
        kernel_trace_kv_enabled() && should_trace_chunk(kv_index, kv_total_chunks, kv_trace_every);
    auto k_chunk_bhld = k_cache_bhld.slice(/*dim=*/2, kv_start, kv_end);
    auto v_chunk_bhld = v_cache_bhld.slice(/*dim=*/2, kv_start, kv_end);
    if (trace_kv_chunk) {
      maybe_trace_kernel_memory("self_attn", "attn_chunk.kv_sliced", k_chunk_bhld, q_start, q_end, q_len_total, kv_start, kv_end, kv_len);
    }
    const int64_t kv_span = kv_end - kv_start;
    torch::Tensor k_chunk = k_chunk_full;
    torch::Tensor v_chunk = v_chunk_full;
    if (kv_span != kv_span_full) {
      TORCH_CHECK(k_chunk_tail.has_value() && v_chunk_tail.has_value(), "streaming_attention_self_chunk_bhld: missing tail buffers");
      k_chunk = *k_chunk_tail;
      v_chunk = *v_chunk_tail;
    }
    k_chunk.copy_(k_chunk_bhld);
    v_chunk.copy_(v_chunk_bhld);
    if (use_cuda_core) {
      launch_streaming_attention_update_fp32(q_chunk, k_chunk, v_chunk, m, l, acc, head_dim);
      if (trace_kv_chunk) {
        maybe_trace_kernel_memory("self_attn", "attn_chunk.cuda_core_update", acc, q_start, q_end, q_len_total, kv_start, kv_end, kv_len);
      }
    } else {
      const auto scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
      auto scores = torch::matmul(q_chunk, k_chunk.transpose(-2, -1));
      if (trace_kv_chunk) {
        maybe_trace_kernel_memory("self_attn", "attn_chunk.scores_matmul", scores, q_start, q_end, q_len_total, kv_start, kv_end, kv_len);
      }
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
    if (trace_kv_chunk) {
      maybe_trace_kernel_memory("self_attn", "attn_chunk.acc_updated", acc, q_start, q_end, q_len_total, kv_start, kv_end, kv_len);
    }
  }

  auto out = (acc / l.clamp_min(1e-9f)).to(output_dtype);
  if (trace_q_chunk) {
    maybe_trace_kernel_memory("self_attn", "attn_chunk.out_ready", out, q_start, q_end, q_len_total, kv_len, kv_len, kv_len);
  }
  return out;
}

torch::Tensor streaming_attention_cross_chunk_bhld(
    const torch::Tensor& q_chunk_bhld,
    const torch::Tensor& k_cache_bhld,
    const torch::Tensor& v_cache_bhld,
    int64_t kv_chunk_size,
    int64_t batch,
    int64_t heads,
    int64_t head_dim,
    AttentionCoreMode core_mode,
    bool trace_q_chunk,
    int64_t q_start,
    int64_t q_end,
    int64_t q_len_total) {
  TORCH_CHECK(q_chunk_bhld.dim() == 4, "streaming_attention_cross_chunk_bhld: q_chunk must be [B,H,Lq,D]");
  TORCH_CHECK(k_cache_bhld.dim() == 4, "streaming_attention_cross_chunk_bhld: k_cache must be [B,H,Lk,D]");
  TORCH_CHECK(v_cache_bhld.dim() == 4, "streaming_attention_cross_chunk_bhld: v_cache must be [B,H,Lk,D]");
  TORCH_CHECK(
      k_cache_bhld.size(0) == batch && k_cache_bhld.size(1) == heads && k_cache_bhld.size(3) == head_dim,
      "streaming_attention_cross_chunk_bhld: k_cache shape mismatch");
  TORCH_CHECK(
      v_cache_bhld.size(0) == batch && v_cache_bhld.size(1) == heads && v_cache_bhld.size(3) == head_dim,
      "streaming_attention_cross_chunk_bhld: v_cache shape mismatch");
  TORCH_CHECK(
      k_cache_bhld.size(2) == v_cache_bhld.size(2),
      "streaming_attention_cross_chunk_bhld: k_cache/v_cache sequence mismatch");
  const int64_t q_span = q_chunk_bhld.size(2);
  const int64_t kv_len = k_cache_bhld.size(2);
  const auto device = q_chunk_bhld.device();
  const auto output_dtype = q_chunk_bhld.scalar_type();
  const auto options_fp32 = q_chunk_bhld.options().dtype(torch::kFloat);
  const bool use_cuda_core = core_mode == AttentionCoreMode::CUDA_STREAMING_EXPERIMENTAL && head_dim <= kCudaCoreMaxHeadDim;
  const int64_t attention_elements = checked_mul_int64(q_len_total, kv_len, "cross_attention_elements");
  if (trace_q_chunk && core_mode == AttentionCoreMode::ATE_NAIVE && attention_elements > kSmallAttentionMatrixElementsBypass) {
    maybe_trace_kernel_memory("cross_attn", "aten_long_seq_path", q_chunk_bhld, q_start, q_end, q_len_total, 0, kv_len, kv_len);
  }
  if (trace_q_chunk && core_mode == AttentionCoreMode::CUDA_STREAMING_EXPERIMENTAL && !use_cuda_core) {
    maybe_trace_kernel_memory("cross_attn", "cuda_core_head_dim_unsupported", q_chunk_bhld, q_start, q_end, q_len_total, 0, kv_len, kv_len);
  }
  auto q_chunk = q_chunk_bhld.to(torch::kFloat).contiguous();
  if (trace_q_chunk) {
    maybe_trace_kernel_memory("cross_attn", "attn_chunk.q_fp32", q_chunk, q_start, q_end, q_len_total, 0, 0, kv_len);
  }
  auto m = torch::full({batch, heads, q_span, 1}, -std::numeric_limits<float>::infinity(), options_fp32.device(device));
  auto l = torch::zeros({batch, heads, q_span, 1}, options_fp32.device(device));
  auto acc = torch::zeros({batch, heads, q_span, head_dim}, options_fp32.device(device));

  const int64_t kv_span_full = std::min<int64_t>(kv_len, kv_chunk_size);
  auto k_chunk_full = torch::empty({batch, heads, kv_span_full, head_dim}, options_fp32.device(device));
  auto v_chunk_full = torch::empty({batch, heads, kv_span_full, head_dim}, options_fp32.device(device));
  const int64_t kv_span_tail = (kv_len > kv_chunk_size) ? (kv_len % kv_chunk_size) : 0;
  c10::optional<torch::Tensor> k_chunk_tail = c10::nullopt;
  c10::optional<torch::Tensor> v_chunk_tail = c10::nullopt;
  if (kv_span_tail != 0) {
    k_chunk_tail = torch::empty({batch, heads, kv_span_tail, head_dim}, options_fp32.device(device));
    v_chunk_tail = torch::empty({batch, heads, kv_span_tail, head_dim}, options_fp32.device(device));
  }

  const int64_t kv_total_chunks = (kv_len + kv_chunk_size - 1) / kv_chunk_size;
  const int64_t kv_trace_every = kernel_trace_every_kv_chunk();
  for (int64_t kv_start = 0; kv_start < kv_len; kv_start += kv_chunk_size) {
    const int64_t kv_end = std::min<int64_t>(kv_len, kv_start + kv_chunk_size);
    const int64_t kv_index = kv_start / kv_chunk_size;
    const bool trace_kv_chunk =
        kernel_trace_kv_enabled() && should_trace_chunk(kv_index, kv_total_chunks, kv_trace_every);
    auto k_chunk_bhld = k_cache_bhld.slice(/*dim=*/2, kv_start, kv_end);
    auto v_chunk_bhld = v_cache_bhld.slice(/*dim=*/2, kv_start, kv_end);
    if (trace_kv_chunk) {
      maybe_trace_kernel_memory("cross_attn", "attn_chunk.kv_sliced", k_chunk_bhld, q_start, q_end, q_len_total, kv_start, kv_end, kv_len);
    }
    const int64_t kv_span = kv_end - kv_start;
    torch::Tensor k_chunk = k_chunk_full;
    torch::Tensor v_chunk = v_chunk_full;
    if (kv_span != kv_span_full) {
      TORCH_CHECK(k_chunk_tail.has_value() && v_chunk_tail.has_value(), "streaming_attention_cross_chunk_bhld: missing tail buffers");
      k_chunk = *k_chunk_tail;
      v_chunk = *v_chunk_tail;
    }
    k_chunk.copy_(k_chunk_bhld);
    v_chunk.copy_(v_chunk_bhld);
    if (use_cuda_core) {
      launch_streaming_attention_update_fp32(q_chunk, k_chunk, v_chunk, m, l, acc, head_dim);
      if (trace_kv_chunk) {
        maybe_trace_kernel_memory("cross_attn", "attn_chunk.cuda_core_update", acc, q_start, q_end, q_len_total, kv_start, kv_end, kv_len);
      }
    } else {
      const auto scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
      auto scores = torch::matmul(q_chunk, k_chunk.transpose(-2, -1));
      if (trace_kv_chunk) {
        maybe_trace_kernel_memory("cross_attn", "attn_chunk.scores_matmul", scores, q_start, q_end, q_len_total, kv_start, kv_end, kv_len);
      }
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
    if (trace_kv_chunk) {
      maybe_trace_kernel_memory("cross_attn", "attn_chunk.acc_updated", acc, q_start, q_end, q_len_total, kv_start, kv_end, kv_len);
    }
  }

  auto out = (acc / l.clamp_min(1e-9f)).to(output_dtype);
  if (trace_q_chunk) {
    maybe_trace_kernel_memory("cross_attn", "attn_chunk.out_ready", out, q_start, q_end, q_len_total, kv_len, kv_len, kv_len);
  }
  return out;
}

torch::Tensor linear_lastdim(
    const torch::Tensor& x_blc,
    const torch::Tensor& w_oi,
    const c10::optional<torch::Tensor>& bias_o) {
  auto x2d = x_blc.contiguous().view({-1, x_blc.size(-1)});
  TORCH_CHECK(w_oi.dim() == 2, "linear_lastdim: weight must be 2D");
  TORCH_CHECK(w_oi.size(1) == x2d.size(1), "linear_lastdim: weight in_features mismatch");
  auto out2d = torch::matmul(x2d, w_oi.transpose(0, 1));
  if (bias_o.has_value()) {
    TORCH_CHECK(bias_o->dim() == 1, "linear_lastdim: bias must be 1D");
    TORCH_CHECK(bias_o->numel() == w_oi.size(0), "linear_lastdim: bias out_features mismatch");
    out2d = out2d + bias_o->view({1, bias_o->size(0)});
  }
  auto out = out2d.view({x_blc.size(0), x_blc.size(1), out2d.size(-1)});
  return out;
}

torch::Tensor project_linear_chunk_bhld(
    const torch::Tensor& source_blc,
    int64_t start,
    int64_t end,
    const torch::Tensor& weight_oi,
    const c10::optional<torch::Tensor>& bias_o,
    int64_t batch,
    int64_t heads,
    int64_t head_dim) {
  TORCH_CHECK(start >= 0 && end >= start && end <= source_blc.size(1), "project_linear_chunk_bhld: invalid chunk range");
  const int64_t span = end - start;
  auto projected_blc = linear_lastdim(source_blc.slice(/*dim=*/1, start, end), weight_oi, bias_o);
  auto projected_blhd = projected_blc.contiguous().view({batch, span, heads, head_dim});
  return projected_blhd.permute({0, 2, 1, 3}).contiguous();
}

torch::Tensor project_norm_rope_chunk_bhld(
    const torch::Tensor& source_blc,
    int64_t start,
    int64_t end,
    const torch::Tensor& weight_oi,
    const c10::optional<torch::Tensor>& bias_o,
    const torch::Tensor& norm_weight,
    const torch::Tensor& rope_cos,
    const torch::Tensor& rope_sin,
    int64_t batch,
    int64_t heads,
    int64_t head_dim) {
  TORCH_CHECK(start >= 0 && end >= start && end <= source_blc.size(1), "project_norm_rope_chunk_bhld: invalid chunk range");
  const int64_t span = end - start;
  auto projected_blc = linear_lastdim(source_blc.slice(/*dim=*/1, start, end), weight_oi, bias_o);
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
    const c10::optional<torch::Tensor>& b_out,
    const c10::optional<std::string>& attn_core) {
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

  const auto bsz = x.size(0);
  const auto seq_len = x.size(1);
  const auto channels = x.size(2);
  TORCH_CHECK(
      rope_cos_qk.dim() == 4 && rope_cos_qk.size(0) == 1 && rope_cos_qk.size(1) == seq_len && rope_cos_qk.size(2) == 1 &&
          rope_cos_qk.size(3) > 0,
      "wan_fused_v1.self_fwd: rope_cos_qk must be [1,L,1,D]");
  TORCH_CHECK(
      rope_sin_qk.dim() == 4 && rope_sin_qk.size(0) == 1 && rope_sin_qk.size(1) == seq_len && rope_sin_qk.size(2) == 1 &&
          rope_sin_qk.size(3) > 0,
      "wan_fused_v1.self_fwd: rope_sin_qk must be [1,L,1,D]");
  const auto head_dim = rope_cos_qk.size(3);
  TORCH_CHECK(rope_sin_qk.size(3) == head_dim, "wan_fused_v1.self_fwd: RoPE head_dim mismatch");
  TORCH_CHECK(channels % head_dim == 0, "wan_fused_v1.self_fwd: channels/head_dim mismatch");
  const auto num_heads = channels / head_dim;
  TORCH_CHECK(w_q.dim() == 2 && w_q.size(0) == channels && w_q.size(1) == channels, "wan_fused_v1.self_fwd: w_q must be [C,C]");
  TORCH_CHECK(w_k.dim() == 2 && w_k.size(0) == channels && w_k.size(1) == channels, "wan_fused_v1.self_fwd: w_k must be [C,C]");
  TORCH_CHECK(w_v.dim() == 2 && w_v.size(0) == channels && w_v.size(1) == channels, "wan_fused_v1.self_fwd: w_v must be [C,C]");
  TORCH_CHECK(w_out.dim() == 2 && w_out.size(0) == channels && w_out.size(1) == channels, "wan_fused_v1.self_fwd: w_out must be [C,C]");
  TORCH_CHECK(norm_q_weight.numel() == channels && norm_k_weight.numel() == channels,
              "wan_fused_v1.self_fwd: norm weights must be [C]");

  const c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t q_chunk_size =
      parse_env_chunk_or_default("CODEX_WAN_FUSED_V1_Q_CHUNK", kDefaultQChunk, kMaxQChunk);
  const int64_t kv_chunk_size =
      parse_env_chunk_or_default("CODEX_WAN_FUSED_V1_KV_CHUNK", kDefaultKvChunk, kMaxKvChunk);

  const int64_t kv_cache_element_bytes =
      std::max<int64_t>(static_cast<int64_t>(x.element_size()),
                        std::max<int64_t>(static_cast<int64_t>(w_k.element_size()), static_cast<int64_t>(w_v.element_size())));

  const bool has_q_bias = b_q.has_value();
  const bool has_k_bias = b_k.has_value();
  const bool has_v_bias = b_v.has_value();
  TORCH_CHECK(
      has_q_bias == has_k_bias && has_q_bias == has_v_bias,
      "wan_fused_v1.self_fwd: requires all-or-none biases for q/k/v.");
  if (has_q_bias) {
    TORCH_CHECK(
        b_q->dim() == 1 && b_q->numel() == channels,
        "wan_fused_v1.self_fwd: b_q must be [C]");
  }
  if (has_k_bias) {
    TORCH_CHECK(
        b_k->dim() == 1 && b_k->numel() == channels,
        "wan_fused_v1.self_fwd: b_k must be [C]");
  }
  if (has_v_bias) {
    TORCH_CHECK(
        b_v->dim() == 1 && b_v->numel() == channels,
        "wan_fused_v1.self_fwd: b_v must be [C]");
  }
  c10::optional<torch::Tensor> bq = has_q_bias ? c10::optional<torch::Tensor>(*b_q) : c10::nullopt;
  c10::optional<torch::Tensor> bk = has_k_bias ? c10::optional<torch::Tensor>(*b_k) : c10::nullopt;
  c10::optional<torch::Tensor> bv = has_v_bias ? c10::optional<torch::Tensor>(*b_v) : c10::nullopt;

  const StreamingPlan plan =
      enforce_streaming_invariants(
          bsz, num_heads, seq_len, seq_len, head_dim, kv_cache_element_bytes, q_chunk_size, kv_chunk_size);
  const int64_t resolved_q_chunk = plan.q_chunk;
  const int64_t resolved_kv_chunk = plan.kv_chunk;
  const int64_t q_total_chunks = (seq_len + resolved_q_chunk - 1) / resolved_q_chunk;
  const int64_t q_trace_every = kernel_trace_every_q_chunk();
  const AttentionCoreMode core_mode = attention_core_mode(attn_core, "wan_fused_v1.self_fwd: attn_core");

  auto k_cache_bhld = project_norm_rope_chunk_bhld(
      x,
      /*start=*/0,
      /*end=*/seq_len,
      w_k,
      bk,
      norm_k_weight,
      rope_cos_qk,
      rope_sin_qk,
      bsz,
      num_heads,
      head_dim);
  auto v_cache_bhld = project_linear_chunk_bhld(
      x,
      /*start=*/0,
      /*end=*/seq_len,
      w_v,
      bv,
      bsz,
      num_heads,
      head_dim);

  auto out = torch::empty({bsz, seq_len, channels}, x.options());
  maybe_trace_kernel_memory("self_attn", "dispatch", x, 0, 0, seq_len, 0, 0, seq_len);
  maybe_trace_kernel_memory("self_attn", "kv_cache.ready", k_cache_bhld, 0, 0, seq_len, 0, seq_len, seq_len);
  for (int64_t q_start = 0; q_start < seq_len; q_start += resolved_q_chunk) {
    const int64_t q_end = std::min<int64_t>(seq_len, q_start + resolved_q_chunk);
    const int64_t q_span = q_end - q_start;
    const int64_t q_index = q_start / resolved_q_chunk;
    const bool trace_q_chunk = should_trace_chunk(q_index, q_total_chunks, q_trace_every);
    if (trace_q_chunk) {
      maybe_trace_kernel_memory("self_attn", "q_chunk.pre", x, q_start, q_end, seq_len, 0, 0, seq_len);
    }
    auto q_chunk_bhld = project_norm_rope_chunk_bhld(
        x,
        q_start,
        q_end,
        w_q,
        bq,
        norm_q_weight,
        rope_cos_qk,
        rope_sin_qk,
        bsz,
        num_heads,
        head_dim);
    if (trace_q_chunk) {
      maybe_trace_kernel_memory("self_attn", "q_chunk.q_projected", q_chunk_bhld, q_start, q_end, seq_len, 0, 0, seq_len);
    }
    auto attn_chunk_bhld =
        streaming_attention_self_chunk_bhld(
            q_chunk_bhld,
            k_cache_bhld,
            v_cache_bhld,
            resolved_kv_chunk,
            bsz,
            num_heads,
            head_dim,
            core_mode,
            trace_q_chunk,
            q_start,
            q_end,
            seq_len);
    auto attn_chunk_blc =
        attn_chunk_bhld.permute({0, 2, 1, 3}).contiguous().view({bsz, q_span, channels});
    out.slice(/*dim=*/1, q_start, q_end).copy_(linear_lastdim(attn_chunk_blc, w_out, b_out));
    if (trace_q_chunk) {
      maybe_trace_kernel_memory("self_attn", "q_chunk.out_written", out, q_start, q_end, seq_len, seq_len, seq_len, seq_len);
    }
  }
  maybe_trace_kernel_memory("self_attn", "complete", out, seq_len, seq_len, seq_len, seq_len, seq_len, seq_len);
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
    const c10::optional<torch::Tensor>& b_out,
    const c10::optional<std::string>& attn_core) {
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

  TORCH_CHECK(norm_q_weight.numel() == channels && norm_k_weight.numel() == channels,
              "wan_fused_v1.cross_fwd: norm weights must be [C]");

  TORCH_CHECK(
      rope_cos_q.dim() == 4 && rope_cos_q.size(0) == 1 && rope_cos_q.size(1) == q_len && rope_cos_q.size(2) == 1 &&
          rope_cos_q.size(3) > 0,
              "wan_fused_v1.cross_fwd: rope_cos_q must be [1,Lq,1,D]");
  TORCH_CHECK(
      rope_sin_q.dim() == 4 && rope_sin_q.size(0) == 1 && rope_sin_q.size(1) == q_len && rope_sin_q.size(2) == 1 &&
          rope_sin_q.size(3) > 0,
              "wan_fused_v1.cross_fwd: rope_sin_q must be [1,Lq,1,D]");
  TORCH_CHECK(
      rope_cos_k.dim() == 4 && rope_cos_k.size(0) == 1 && rope_cos_k.size(1) == kv_len && rope_cos_k.size(2) == 1 &&
          rope_cos_k.size(3) > 0,
              "wan_fused_v1.cross_fwd: rope_cos_k must be [1,Lk,1,D]");
  TORCH_CHECK(
      rope_sin_k.dim() == 4 && rope_sin_k.size(0) == 1 && rope_sin_k.size(1) == kv_len && rope_sin_k.size(2) == 1 &&
          rope_sin_k.size(3) > 0,
              "wan_fused_v1.cross_fwd: rope_sin_k must be [1,Lk,1,D]");
  const auto head_dim = rope_cos_q.size(3);
  TORCH_CHECK(rope_sin_q.size(3) == head_dim, "wan_fused_v1.cross_fwd: RoPE head_dim mismatch (q)");
  TORCH_CHECK(rope_cos_k.size(3) == head_dim, "wan_fused_v1.cross_fwd: RoPE head_dim mismatch (k_cos)");
  TORCH_CHECK(rope_sin_k.size(3) == head_dim, "wan_fused_v1.cross_fwd: RoPE head_dim mismatch (k_sin)");
  TORCH_CHECK(channels % head_dim == 0, "wan_fused_v1.cross_fwd: channels/head_dim mismatch");
  const auto num_heads = channels / head_dim;
  TORCH_CHECK(w_q.dim() == 2 && w_q.size(0) == channels && w_q.size(1) == channels, "wan_fused_v1.cross_fwd: w_q must be [C,C]");
  TORCH_CHECK(w_k.dim() == 2 && w_k.size(0) == channels && w_k.size(1) == ctx_dim, "wan_fused_v1.cross_fwd: w_k must be [C,Cctx]");
  TORCH_CHECK(w_v.dim() == 2 && w_v.size(0) == channels && w_v.size(1) == ctx_dim, "wan_fused_v1.cross_fwd: w_v must be [C,Cctx]");
  TORCH_CHECK(w_out.dim() == 2 && w_out.size(0) == channels && w_out.size(1) == channels, "wan_fused_v1.cross_fwd: w_out must be [C,C]");

  const c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t q_chunk_size =
      parse_env_chunk_or_default("CODEX_WAN_FUSED_V1_Q_CHUNK", kDefaultQChunk, kMaxQChunk);
  const int64_t kv_chunk_size =
      parse_env_chunk_or_default("CODEX_WAN_FUSED_V1_KV_CHUNK", kDefaultKvChunk, kMaxKvChunk);

  c10::optional<torch::Tensor> bq_flat = c10::nullopt;
  c10::optional<torch::Tensor> bk_flat = c10::nullopt;
  c10::optional<torch::Tensor> bv_flat = c10::nullopt;
  if (b_q.has_value()) {
    TORCH_CHECK(b_q->dim() == 1 && b_q->numel() == channels,
                "wan_fused_v1.cross_fwd: b_q must be [C]");
    bq_flat = *b_q;
  }
  if (b_k.has_value()) {
    TORCH_CHECK(b_k->dim() == 1 && b_k->numel() == channels,
                "wan_fused_v1.cross_fwd: b_k must be [C]");
    bk_flat = *b_k;
  }
  if (b_v.has_value()) {
    TORCH_CHECK(b_v->dim() == 1 && b_v->numel() == channels,
                "wan_fused_v1.cross_fwd: b_v must be [C]");
    bv_flat = *b_v;
  }
  if (q_len == 0) {
    return torch::empty({bsz, q_len, channels}, x.options());
  }
  const int64_t kv_cache_element_bytes =
      std::max<int64_t>(static_cast<int64_t>(context.element_size()),
                        std::max<int64_t>(static_cast<int64_t>(w_k.element_size()), static_cast<int64_t>(w_v.element_size())));

  const StreamingPlan plan =
      enforce_streaming_invariants(
          bsz, num_heads, q_len, kv_len, head_dim, kv_cache_element_bytes, q_chunk_size, kv_chunk_size);
  const int64_t resolved_q_chunk = plan.q_chunk;
  const int64_t resolved_kv_chunk = plan.kv_chunk;
  const int64_t q_total_chunks = (q_len + resolved_q_chunk - 1) / resolved_q_chunk;
  const int64_t q_trace_every = kernel_trace_every_q_chunk();
  const AttentionCoreMode core_mode = attention_core_mode(attn_core, "wan_fused_v1.cross_fwd: attn_core");

  auto k_cache_bhld = project_norm_rope_chunk_bhld(
      context,
      /*start=*/0,
      /*end=*/kv_len,
      w_k,
      bk_flat,
      norm_k_weight,
      rope_cos_k,
      rope_sin_k,
      bsz,
      num_heads,
      head_dim);
  auto v_cache_bhld = project_linear_chunk_bhld(
      context,
      /*start=*/0,
      /*end=*/kv_len,
      w_v,
      bv_flat,
      bsz,
      num_heads,
      head_dim);

  auto out = torch::empty({bsz, q_len, channels}, x.options());
  maybe_trace_kernel_memory("cross_attn", "dispatch", x, 0, 0, q_len, 0, 0, kv_len);
  maybe_trace_kernel_memory("cross_attn", "kv_cache.ready", k_cache_bhld, 0, 0, q_len, 0, kv_len, kv_len);
  for (int64_t q_start = 0; q_start < q_len; q_start += resolved_q_chunk) {
    const int64_t q_end = std::min<int64_t>(q_len, q_start + resolved_q_chunk);
    const int64_t q_span = q_end - q_start;
    const int64_t q_index = q_start / resolved_q_chunk;
    const bool trace_q_chunk = should_trace_chunk(q_index, q_total_chunks, q_trace_every);
    if (trace_q_chunk) {
      maybe_trace_kernel_memory("cross_attn", "q_chunk.pre", x, q_start, q_end, q_len, 0, 0, kv_len);
    }
    auto q_chunk_bhld = project_norm_rope_chunk_bhld(
        x,
        q_start,
        q_end,
        w_q,
        bq_flat,
        norm_q_weight,
        rope_cos_q,
        rope_sin_q,
        bsz,
        num_heads,
        head_dim);
    if (trace_q_chunk) {
      maybe_trace_kernel_memory("cross_attn", "q_chunk.q_projected", q_chunk_bhld, q_start, q_end, q_len, 0, 0, kv_len);
    }
    auto attn_chunk_bhld =
        streaming_attention_cross_chunk_bhld(
            q_chunk_bhld,
            k_cache_bhld,
            v_cache_bhld,
            resolved_kv_chunk,
            bsz,
            num_heads,
            head_dim,
            core_mode,
            trace_q_chunk,
            q_start,
            q_end,
            q_len);
    auto attn_chunk_blc =
        attn_chunk_bhld.permute({0, 2, 1, 3}).contiguous().view({bsz, q_span, channels});
    out.slice(/*dim=*/1, q_start, q_end).copy_(linear_lastdim(attn_chunk_blc, w_out, b_out));
    if (trace_q_chunk) {
      maybe_trace_kernel_memory("cross_attn", "q_chunk.out_written", out, q_start, q_end, q_len, kv_len, kv_len, kv_len);
    }
  }
  maybe_trace_kernel_memory("cross_attn", "complete", out, q_len, q_len, q_len, kv_len, kv_len, kv_len);
  return out;
}
