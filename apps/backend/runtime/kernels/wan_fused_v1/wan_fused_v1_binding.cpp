// WAN fused attention V1 C++/CUDA extension bindings.
// Build with: python setup.py build_ext --inplace (requires CUDA toolchain).

#include <torch/extension.h>

#include <string>

namespace {
constexpr int kWanFusedV1AbiVersion = 4;
}

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
    const c10::optional<std::string>& attn_core);

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
    const c10::optional<std::string>& attn_core);

TORCH_LIBRARY(wan_fused_v1, m) {
  m.def(
      "self_fwd(Tensor x, Tensor w_q, Tensor? b_q, Tensor w_k, Tensor? b_k, Tensor w_v, Tensor? b_v, Tensor norm_q_weight, Tensor norm_k_weight, Tensor rope_cos_qk, Tensor rope_sin_qk, Tensor w_out, Tensor? b_out, str? attn_core=None) -> Tensor");
  m.def(
      "cross_fwd(Tensor x, Tensor context, Tensor w_q, Tensor? b_q, Tensor norm_q_weight, Tensor rope_cos_q, Tensor rope_sin_q, Tensor w_k, Tensor? b_k, Tensor norm_k_weight, Tensor rope_cos_k, Tensor rope_sin_k, Tensor w_v, Tensor? b_v, Tensor w_out, Tensor? b_out, str? attn_core=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(wan_fused_v1, CPU, m) {
  m.impl(
      "self_fwd",
      [](const torch::Tensor& x,
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
         const c10::optional<std::string>& /*attn_core*/) -> torch::Tensor {
        TORCH_CHECK(
            false,
            "wan_fused_v1.self_fwd: CPU implementation not available. Build CUDA kernels and run on CUDA tensors.");
      });

  m.impl(
      "cross_fwd",
      [](const torch::Tensor& x,
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
         const c10::optional<std::string>& /*attn_core*/) -> torch::Tensor {
        TORCH_CHECK(
            false,
            "wan_fused_v1.cross_fwd: CPU implementation not available. Build CUDA kernels and run on CUDA tensors.");
      });
}

TORCH_LIBRARY_IMPL(wan_fused_v1, CUDA, m) {
  m.impl("self_fwd", wan_fused_v1_self_fwd_cuda);
  m.impl("cross_fwd", wan_fused_v1_cross_fwd_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("WAN_FUSED_V1_ABI") = kWanFusedV1AbiVersion;
  m.def(
      "self_fwd",
      [](const torch::Tensor& x,
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
         const c10::optional<torch::Tensor>& b_out) -> torch::Tensor {
        return wan_fused_v1_self_fwd_cuda(
            x,
            w_q,
            b_q,
            w_k,
            b_k,
            w_v,
            b_v,
            norm_q_weight,
            norm_k_weight,
            rope_cos_qk,
            rope_sin_qk,
            w_out,
            b_out,
            c10::nullopt);
      },
      "WAN fused V1 self attention forward (CUDA, legacy ABI)");
  m.def("self_fwd", &wan_fused_v1_self_fwd_cuda, "WAN fused V1 self attention forward (CUDA)");
  m.def(
      "cross_fwd",
      [](const torch::Tensor& x,
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
         const c10::optional<torch::Tensor>& b_out) -> torch::Tensor {
        return wan_fused_v1_cross_fwd_cuda(
            x,
            context,
            w_q,
            b_q,
            norm_q_weight,
            rope_cos_q,
            rope_sin_q,
            w_k,
            b_k,
            norm_k_weight,
            rope_cos_k,
            rope_sin_k,
            w_v,
            b_v,
            w_out,
            b_out,
            c10::nullopt);
      },
      "WAN fused V1 cross attention forward (CUDA, legacy ABI)");
  m.def("cross_fwd", &wan_fused_v1_cross_fwd_cuda, "WAN fused V1 cross attention forward (CUDA)");
}
