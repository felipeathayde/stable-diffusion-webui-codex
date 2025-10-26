// Skeleton CUDA kernel for FP8 linear (matmul + bias) with per-output scale.
// Placeholder: for now, dequantize to FP16 on-the-fly via aten to keep pipeline wired.

#include <torch/extension.h>
using torch::Tensor;

Tensor te_linear_fp8_forward(const Tensor& x, const Tensor& w_u8, const Tensor& w_scale,
                             const c10::optional<Tensor>& b, const int8_t fp8_format) {
  TORCH_CHECK(x.is_cuda() && w_u8.is_cuda() && w_scale.is_cuda(), "linear_fp8_forward: tensors must be CUDA");
  TORCH_CHECK(x.dim()==3, "linear_fp8_forward: x expected [B,L,Cin]");
  // Placeholder: treat w_u8 as fp16 weights via fake dequant (u8 * scale)
  auto dtype = x.dtype();
  Tensor w = w_u8.to(torch::kFloat32).mul(w_scale.view({-1,1}));
  w = w.to(dtype);
  Tensor out = torch::matmul(x, w.transpose(0,1));
  if (b.has_value()) out.add_(b.value());
  return out;
}

