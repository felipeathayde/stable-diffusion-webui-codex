// Experimental CUDA kernel: streaming softmax attention per query row chunk.
// This avoids materializing full [Lq,Lk] logits. Two-pass algorithm per row:
// 1) find max over K tiles; 2) sum exp((q·k)/sqrt(D) - max); 3) compute output as sum(exp(...) * v) / sumexp.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
__device__ inline T my_max(T a, T b) { return a > b ? a : b; }

// Kernel assumes: Q [B*H, Lq, D], K [B*H, Lk, D], V [B*H, Lk, D]
// Each block handles one row (one query position) within a chunk for a specific batch-head.
template <typename scalar_t>
__global__ void attn_stream_rows_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ Out,
    const int BxH, const int Lq, const int Lk, const int D,
    const int q_chunk_start, const int q_chunk_len,
    const float scale)
{
    const int bh = blockIdx.z;            // [0, B*H)
    const int q_row = q_chunk_start + blockIdx.y; // absolute row in [0, Lq)
    if (q_row >= Lq) return;
    const int tid = threadIdx.x;          // along D

    const int row_offset_q = (bh*Lq + q_row)*D;
    // First pass: find max logit over all keys
    float max_logit = -1e30f;
    for (int k_row = 0; k_row < Lk; ++k_row) {
        // dot(q_row, k_row)
        float dot = 0.f;
        for (int d = tid; d < D; d += blockDim.x) {
            dot += static_cast<float>(Q[row_offset_q + d]) * static_cast<float>(K[(bh*Lk + k_row)*D + d]);
        }
        // reduce within block
        __shared__ float shm[1024/32]; // assuming max blockDim.x 1024; we keep it simple
        float val = dot;
        // warp reduce
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) == 0) shm[threadIdx.x >> 5] = val;
        __syncthreads();
        if (threadIdx.x < 32) {
            float sum = (threadIdx.x < (blockDim.x>>5)) ? shm[threadIdx.x] : 0.f;
            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            if (threadIdx.x == 0) {
                float logit = sum * scale;
                max_logit = my_max(max_logit, logit);
            }
        }
        __syncthreads();
    }

    // Broadcast max_logit within block
    max_logit = __shfl_sync(0xffffffff, max_logit, 0);

    // Second pass: sumexp and output accumulation
    float denom = 0.f;
    // We accumulate output vector per thread over D (partial), then reduce
    extern __shared__ float out_partial[]; // size = blockDim.x
    float outacc = 0.f; // per-d element handled by this thread will be reduced in a second phase per D; simplify to scalar reduce by iterating D per thread

    // We'll accumulate Out row in global memory with atomic adds (simplify). For better perf, use shared tile [D] and write once.
    for (int dstart = 0; dstart < D; dstart += blockDim.x) {
        // temp buffer for partial sums per D element
        int d = dstart + threadIdx.x;
        float out_d = 0.f;
        for (int k_row = 0; k_row < Lk; ++k_row) {
            // dot(q_row,k_row) recompute (we could cache partial, kept simple)
            float dot = 0.f;
            for (int t = dstart; t < D; t += blockDim.x) {
                // In this loop, we compute dot redundantly per d window; acceptable for simplicity in first version
                // For correctness, we need exp(logit - max)
            }
        }
    }
    // NOTE: For brevity and safety, we keep the kernel scaffold minimal here; real implementation will replace
}

// Launcher stub (keeps API while we iterate)
std::vector<torch::Tensor> attn_stream_rows_launcher(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const bool causal, const int64_t chunk)
{
    // For now, fall back to ATen chunked path in launcher to keep behavior stable until kernel completes.
    auto B = q.size(0); auto H = q.size(1); auto L = q.size(2); auto D = q.size(3);
    auto qv = q.reshape({B*H, L, D});
    auto kv = k.reshape({B*H, L, D});
    auto vv = v.reshape({B*H, L, D});
    auto kT = kv.transpose(1,2);
    auto out = torch::empty_like(qv);
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    for (int64_t s = 0; s < L; s += chunk) {
        int64_t c = std::min<int64_t>(chunk, L - s);
        auto q_chunk = qv.narrow(1, s, c);
        auto logits = torch::matmul(q_chunk, kT) * scale;
        if (causal) {
            // rudimentary causal row-wise mask for current chunk
            auto arL = torch::arange(L, q.options().device(q.device()));
            for (int64_t i = 0; i < c; ++i) {
                int64_t pos = s + i;
                auto row = logits.select(1, i);
                auto mask = arL.gt(pos).to(row.dtype()) * (-1e9);
                row.add_(mask);
            }
        }
        auto m = std::get<0>(logits.max(-1, true));
        auto p = (logits - m).softmax(-1);
        auto o_chunk = torch::matmul(p, vv);
        out.narrow(1, s, c).copy_(o_chunk);
    }
    return {out.reshape_as(q), torch::Tensor()};
}

