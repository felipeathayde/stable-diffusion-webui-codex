// Minimal PyTorch C++/CUDA extension bindings for WAN T5 FP8 kernels
// NOTE: This is a scaffold. Actual kernels live in te_attention_fp8.cu / te_linear_fp8.cu
// Build with: python setup.py build_ext --inplace (Windows supported if toolchain is present)

#include <torch/extension.h>

// Forward declarations (CUDA implementations)
torch::Tensor te_linear_fp8_forward(
    const torch::Tensor& x,           // [B,L,Cin] fp16/bf16
    const torch::Tensor& w_u8,        // [Cout,Cin] uint8 (FP8 data)
    const torch::Tensor& w_scale,     // [Cout] or [1] float (per-output scale)
    const c10::optional<torch::Tensor>& b, // [Cout] fp16/bf16/float
    const int8_t fp8_format            // 0=e4m3fn, 1=e5m2 (placeholder)
);

std::vector<torch::Tensor> te_attn_fp8_forward(
    const torch::Tensor& q,           // [B,H,L,D]
    const torch::Tensor& k,           // [B,H,L,D]
    const torch::Tensor& v,           // [B,H,L,D]
    const c10::optional<torch::Tensor>& attn_mask, // [B,1,L,L] or null
    const bool causal,
    const int8_t fp8_format
);

TORCH_LIBRARY(wan_te_cuda, m) {
  m.def("linear_fp8_forward(Tensor x, Tensor w_u8, Tensor w_scale, Tensor? b, int fp8_format) -> Tensor");
  m.def("attn_fp8_forward(Tensor q, Tensor k, Tensor v, Tensor? attn_mask, bool causal, int fp8_format) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(wan_te_cuda, CPU, m) {
  m.impl("linear_fp8_forward", [](const torch::Tensor& x,
                                   const torch::Tensor& w_u8,
                                   const torch::Tensor& w_scale,
                                   const c10::optional<torch::Tensor>& b,
                                   int fp8_format) {
    TORCH_CHECK(false, "wan_te_cuda.linear_fp8_forward: CPU implementation not available. Build CUDA kernels.");
  });
  m.impl("attn_fp8_forward", [](const torch::Tensor& q,
                                 const torch::Tensor& k,
                                 const torch::Tensor& v,
                                 const c10::optional<torch::Tensor>& attn_mask,
                                 bool causal,
                                 int fp8_format) {
    TORCH_CHECK(false, "wan_te_cuda.attn_fp8_forward: CPU implementation not available. Build CUDA kernels.");
  });
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(wan_te_cuda, CUDA, m) {
  m.impl("linear_fp8_forward", te_linear_fp8_forward);
  m.impl("attn_fp8_forward", te_attn_fp8_forward);
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_fp8_forward", &te_linear_fp8_forward, "WAN T5 Linear FP8 forward");
  m.def("attn_fp8_forward", &te_attn_fp8_forward, "WAN T5 Attention FP8 forward");
}

