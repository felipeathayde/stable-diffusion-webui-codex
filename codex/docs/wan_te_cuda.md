WAN T5 CUDA FP8 (Experimental)

Goals
- Keep T5 (UMT5‑XXL) weights in FP8 (uint8 + per‑tensor/per‑row scale), avoiding fp16/bf16 upcast in VRAM.
- Compute in fp16/bf16 with on‑the‑fly dequant per tile; stage one block at a time.
- No silent fallbacks: explicit errors when the kernel is unavailable or unsupported.

Compatibility
- Target: PyTorch 2.9.0 + CUDA 12.8. GPUs SM≥70 (weights FP8 + compute fp16) and SM≥90 (optional compute FP8 later).
- Windows: requires Visual Studio 2022 Build Tools (MSVC v143) and CUDA 12.8. Linux: nvcc 12.8.

Build
- In‑place build:
  - cd apps/server/backend/runtime/kernels/wan_t5
  - python setup.py build_ext --inplace
  - Loader looks here first (sys.path), then JIT.
- JIT (one‑shot, cached): set WAN_TE_BUILD=1 e use te_impl='cuda_fp8'.

Pinned memory (Windows/Linux)
- Por padrão o encoder usa host pinned para copiar embeddings FP8 para a GPU. Isso é suportado no Windows e Linux.
- Pode desativar com WAN_TE_PINNED=0 se o driver/ambiente tiver problemas ou se preferir cópias síncronas.
- A cópia continua correta sem pinned; apenas perde latência/overlap.

Runtime Flags
- gguf_te_impl: 'cuda_fp8' | 'hf' | 'cpu'
- gguf_te_kernel_required: true|false (error hard se não houver kernel)
- WAN_TE_TILE: tile de Cout (default 256) para linear FP8.
- WAN_TE_ATTN_CHUNK: tile de queries (default 192) na atenção.
- WAN_TE_ATTN_KCHUNK: tile de chaves (default 256) na atenção streaming.
- WAN_TE_ATTN: 'cuda' | 'sdpa' (auto usa 'cuda' quando a extensão está disponível).
- WAN_TE_ATTN_IMPL: 'kernel' | 'aten' (default 'kernel' quando a extensão está presente).
- WAN_TE_PINNED: 1|0 (default 1) — ativa/desativa host pinned no embedding.

Status
- P1 (Linear FP8): done (tile dequant → GEMM) — correto e com baixo pico de VRAM.
- P2 (Attention FP8): streaming-softmax chunked (launcher) — sem L×L materializado.
- P3 (Encoder): integrado com Linear FP8 + atenção (SDPA/cuda) — em evolução.
