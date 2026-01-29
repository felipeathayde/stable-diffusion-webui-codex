// Minimal PyTorch C++/CUDA extension bindings for CodexPack packed GGUF kernels.
// Build with: python setup.py build_ext --inplace (requires a CUDA toolchain).

#include <torch/extension.h>

torch::Tensor q4k_tilepack_linear_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed,
    int64_t out_features,
    int64_t in_features,
    const c10::optional<torch::Tensor>& bias);

TORCH_LIBRARY(codexpack, m) {
  m.def("q4k_tilepack_linear(Tensor x, Tensor packed, int out_features, int in_features, Tensor? bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(codexpack, CPU, m) {
  m.impl("q4k_tilepack_linear",
         [](const torch::Tensor& x,
            const torch::Tensor& packed,
            int64_t out_features,
            int64_t in_features,
            const c10::optional<torch::Tensor>& bias) -> torch::Tensor {
           TORCH_CHECK(false,
                       "codexpack.q4k_tilepack_linear: CPU implementation not available. Build CUDA kernels.");
         });
}

TORCH_LIBRARY_IMPL(codexpack, CUDA, m) { m.impl("q4k_tilepack_linear", q4k_tilepack_linear_cuda); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("q4k_tilepack_linear",
        &q4k_tilepack_linear_cuda,
        "CodexPack Q4_K tilepack_v1 linear forward (CUDA)");
}
