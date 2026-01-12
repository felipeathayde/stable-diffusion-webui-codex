// Linear FP8 (uint8 + per-output scale) → compute in FP16/BF16 without materializing full W.
// Strategy: tile Cout dimension, dequantize one tile [Cout_t, Cin] to compute dtype, GEMM x@[W_t]^T, accumulate into y[:, :, Cout_tile].

#include <torch/extension.h>
#include <ATen/ATen.h>

using torch::Tensor;

namespace {

inline Tensor dequant_tile(const Tensor& w_u8, const Tensor& w_scale, int64_t r0, int64_t r1, at::ScalarType dtype, c10::Device dev) {
  // w_u8: [Cout, Cin] uint8, w_scale: [Cout] or broadcastable
  auto rows = r1 - r0;
  auto cols = w_u8.size(1);
  Tensor u8_tile = w_u8.narrow(0, r0, rows);                  // [rows, Cin]
  Tensor f32 = u8_tile.to(torch::kFloat32);                    // convert small tile (stays on src device)
  Tensor s = w_scale.narrow(0, r0, rows).to(torch::kFloat32);  // [rows]
  f32.mul_(s.view({rows, 1}));
  return f32.to(dev, dtype);                                   // move to compute device+dtype
}

} // namespace

Tensor te_linear_fp8_forward(const Tensor& x, const Tensor& w_u8, const Tensor& w_scale,
                             const c10::optional<Tensor>& b, const int8_t /*fp8_format*/) {
  TORCH_CHECK(x.is_cuda(), "linear_fp8_forward: input must be CUDA");
  TORCH_CHECK(w_u8.device().is_cpu() || w_u8.is_cuda(), "linear_fp8_forward: w_u8 must be CPU or CUDA");
  TORCH_CHECK(w_scale.device().is_cpu() || w_scale.is_cuda(), "linear_fp8_forward: w_scale must be CPU or CUDA");
  TORCH_CHECK(x.dim()==3, "linear_fp8_forward: x expected [B,L,Cin]");
  TORCH_CHECK(w_u8.dim()==2, "linear_fp8_forward: w_u8 expected [Cout,Cin]");
  TORCH_CHECK(w_scale.dim()==1 || (w_scale.dim()==0), "linear_fp8_forward: w_scale expected [Cout] or scalar");

  auto B = x.size(0);
  auto L = x.size(1);
  auto Cin = x.size(2);
  auto Cout = w_u8.size(0);
  TORCH_CHECK(w_u8.size(1) == Cin, "linear_fp8_forward: Cin mismatch");
  if (w_scale.dim()==1) {
    TORCH_CHECK(w_scale.size(0) == Cout, "linear_fp8_forward: scale size mismatch");
  }

  // Output tensor
  auto dtype = x.dtype();
  Tensor y = torch::zeros({B, L, Cout}, x.options());

  // Tile size (fixed; env overrides removed)
  int64_t tile = 256;

  // Process Cout in tiles
  auto dev = x.device();
  for (int64_t r0 = 0; r0 < Cout; r0 += tile) {
    int64_t r1 = std::min<int64_t>(Cout, r0 + tile);
    // Dequantize tile to compute dtype without materializing full W
    Tensor w_t = dequant_tile(w_u8, w_scale, r0, r1, dtype, dev);   // [rows, Cin]
    // GEMM: [B,L,Cin] @ [rows,Cin]^T -> [B,L,rows]
    // We reshape to 2D for matmul and back to 3D for slice assign
    Tensor x2d = x.reshape({B*L, Cin});
    Tensor y2d = torch::matmul(x2d, w_t.transpose(0,1));       // [B*L, rows]
    Tensor y_tile = y2d.reshape({B, L, r1 - r0});
    if (b.has_value()) {
      Tensor b_slice = b.value().narrow(0, r0, r1 - r0).to(dtype);
      y_tile.add_(b_slice.view({1,1,-1}));
    }
    // write into y
    y.narrow(2, r0, r1 - r0).copy_(y_tile);
  }

  return y;
}
