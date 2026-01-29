// CUDA implementation for CodexPack `cuda.ggml_q4_k.linear.tilepack_v1`.
//
// v1 goals:
// - Correctness-first fused dequant(block) + dot product (no FP16 weight materialization to HBM).
// - SM86+ runtime gate (policy).
// - fp16 activations + fp16 output (policy).
//
// Note: This initial kernel is intentionally simple and is not expected to match highly tuned GEMM
// kernels yet. Performance work (Tensor Cores, pipelining, epilogue fusion) is a v2 follow-up.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kTileM = 128;
constexpr int kTileK = 256;
constexpr int kQ4KBlockBytes = 144;

__global__ void q4k_tilepack_linear_kernel(
    const half* __restrict__ x,              // [M,K]
    const uint8_t* __restrict__ packed_w,    // packed bytes
    half* __restrict__ out,                  // [M,N]
    const half* __restrict__ bias,           // [N] or null
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t k_tiles) {
  // Policy: SM86+ only (checked on the host). Do not silently fall back on older GPUs.

  const int tid = threadIdx.x;
  if (tid >= kTileM) {
    return;
  }

  const int64_t m = static_cast<int64_t>(blockIdx.x);
  const int64_t n_tile = static_cast<int64_t>(blockIdx.y);
  const int64_t n = n_tile * kTileM + tid;
  if (m >= M || n >= N) {
    return;
  }

  extern __shared__ half x_sh[];

  const half* x_row = x + m * K;

  float acc = 0.0f;

  for (int64_t kt = 0; kt < k_tiles; ++kt) {
    // Load activation tile (256 fp16) once per block.
    const int base = static_cast<int>(kt * kTileK);
    const int i0 = tid * 2;
    if (i0 + 1 < kTileK) {
      x_sh[i0] = x_row[base + i0];
      x_sh[i0 + 1] = x_row[base + i0 + 1];
    }
    __syncthreads();

    const int64_t tile_base =
        (n_tile * k_tiles + kt) * (static_cast<int64_t>(kTileM) * kQ4KBlockBytes);
    const uint8_t* block = packed_w + tile_base + static_cast<int64_t>(tid) * kQ4KBlockBytes;

    // Q4_K block layout (144 bytes):
    // - d (fp16): 2
    // - dmin (fp16): 2
    // - scales: 12 (3x4)
    // - qs: 128 (nibble-packed)
    const half d_h = *reinterpret_cast<const half*>(block + 0);
    const half dmin_h = *reinterpret_cast<const half*>(block + 2);

    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;

    // Extract sc[8] and m[8] as per `apps/backend/quantization/dequant.py:get_scale_min`.
    uint8_t sc[8];
    uint8_t mn[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint8_t d_b = scales[i + 0];
      const uint8_t m_b = scales[i + 4];
      const uint8_t md_b = scales[i + 8];
      sc[i] = d_b & 0x3F;
      sc[i + 4] = (md_b & 0x0F) | ((d_b >> 2) & 0x30);
      mn[i] = m_b & 0x3F;
      mn[i + 4] = (md_b >> 4) | ((m_b >> 2) & 0x30);
    }

    half d_sc[8];
    half dm[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
      const half sc_h = __float2half_rn(static_cast<float>(sc[g]));
      const half mn_h = __float2half_rn(static_cast<float>(mn[g]));
      d_sc[g] = __hmul(d_h, sc_h);
      dm[g] = __hmul(dmin_h, mn_h);
    }

    // Dot product over 256 values, organized as 8 groups of 32.
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      const int g0 = chunk * 2;
      const int g1 = g0 + 1;
      const half a0 = d_sc[g0];
      const half b0 = dm[g0];
      const half a1 = d_sc[g1];
      const half b1 = dm[g1];
      const int qs_base = chunk * 32;
      const int x_base0 = g0 * 32;
      const int x_base1 = g1 * 32;
      #pragma unroll
      for (int j = 0; j < 32; ++j) {
        const uint8_t qbyte = qs[qs_base + j];
        const uint8_t q0 = qbyte & 0x0F;
        const uint8_t q1 = qbyte >> 4;
        const half q0_h = __float2half_rn(static_cast<float>(q0));
        const half q1_h = __float2half_rn(static_cast<float>(q1));
        const half w0_h = __hsub(__hmul(a0, q0_h), b0);
        const half w1_h = __hsub(__hmul(a1, q1_h), b1);
        const float x0 = __half2float(x_sh[x_base0 + j]);
        const float x1 = __half2float(x_sh[x_base1 + j]);
        acc = fmaf(__half2float(w0_h), x0, acc);
        acc = fmaf(__half2float(w1_h), x1, acc);
      }
    }

    __syncthreads();
  }

  if (bias != nullptr) {
    acc += __half2float(bias[n]);
  }
  out[m * N + n] = __float2half_rn(acc);
}

}  // namespace

torch::Tensor q4k_tilepack_linear_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed,
    int64_t out_features,
    int64_t in_features,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(x.is_cuda(), "codexpack.q4k_tilepack_linear: x must be CUDA");
  TORCH_CHECK(packed.is_cuda(), "codexpack.q4k_tilepack_linear: packed must be CUDA");
  TORCH_CHECK(x.get_device() == packed.get_device(),
              "codexpack.q4k_tilepack_linear: x and packed must be on the same CUDA device (x=",
              x.get_device(),
              " packed=",
              packed.get_device(),
              ")");
  TORCH_CHECK(x.dtype() == torch::kFloat16, "codexpack.q4k_tilepack_linear: x must be float16");
  TORCH_CHECK(in_features > 0 && out_features > 0, "codexpack.q4k_tilepack_linear: invalid dims");
  TORCH_CHECK(in_features % kTileK == 0, "codexpack.q4k_tilepack_linear: in_features must be multiple of 256");
  TORCH_CHECK(out_features % kTileM == 0, "codexpack.q4k_tilepack_linear: out_features must be multiple of 128");
  TORCH_CHECK(packed.is_contiguous(), "codexpack.q4k_tilepack_linear: packed must be contiguous");
  TORCH_CHECK(packed.scalar_type() == torch::kInt8 || packed.scalar_type() == torch::kUInt8,
              "codexpack.q4k_tilepack_linear: packed must be int8/uint8");

  const c10::cuda::CUDAGuard device_guard(x.device());

  const auto x_contig = x.contiguous();
  const int64_t K = in_features;
  TORCH_CHECK(x_contig.numel() % K == 0, "codexpack.q4k_tilepack_linear: x shape mismatch");
  const int64_t M = x_contig.numel() / K;
  const int64_t k_tiles = K / kTileK;

  // Validate packed blob size.
  const int64_t expected_bytes = out_features * k_tiles * kQ4KBlockBytes;
  TORCH_CHECK(packed.numel() == expected_bytes,
              "codexpack.q4k_tilepack_linear: packed blob has unexpected size. expected ",
              expected_bytes, " bytes, got ", packed.numel());

  // Bias (optional).
  const half* bias_ptr = nullptr;
  torch::Tensor bias_contig;
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(), "codexpack.q4k_tilepack_linear: bias must be CUDA when provided");
    TORCH_CHECK(bias->get_device() == x.get_device(),
                "codexpack.q4k_tilepack_linear: bias must be on the same CUDA device as x (x=",
                x.get_device(),
                " bias=",
                bias->get_device(),
                ")");
    TORCH_CHECK(bias->dtype() == torch::kFloat16, "codexpack.q4k_tilepack_linear: bias must be float16");
    TORCH_CHECK(bias->numel() == out_features, "codexpack.q4k_tilepack_linear: bias size mismatch");
    bias_contig = bias->contiguous();
    bias_ptr = reinterpret_cast<const half*>(bias_contig.data_ptr<at::Half>());
  }

  // Policy: SM86+ only.
  const auto* props = at::cuda::getCurrentDeviceProperties();
  const int major = props->major;
  const int minor = props->minor;
  TORCH_CHECK((major > 8) || (major == 8 && minor >= 6),
              "codexpack.q4k_tilepack_linear: requires SM86+ (got compute_capability=",
              major, ".", minor, ")");

  // Output shape matches torch.nn.Linear: [..., out_features]
  std::vector<int64_t> out_sizes(x_contig.sizes().vec());
  out_sizes.back() = out_features;
  auto out = torch::empty(out_sizes, x_contig.options());
  auto out2d = out.view({M, out_features});

  const int64_t n_tiles = out_features / kTileM;
  const dim3 grid(static_cast<uint32_t>(M), static_cast<uint32_t>(n_tiles), 1);
  const dim3 block(kTileM, 1, 1);
  const int shared_bytes = kTileK * sizeof(half);

  const uint8_t* packed_ptr = reinterpret_cast<const uint8_t*>(packed.data_ptr());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  q4k_tilepack_linear_kernel<<<grid, block, shared_bytes, stream.stream()>>>(
      reinterpret_cast<const half*>(x_contig.data_ptr<at::Half>()),
      packed_ptr,
      reinterpret_cast<half*>(out2d.data_ptr<at::Half>()),
      bias_ptr,
      M,
      out_features,
      K,
      k_tiles);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
