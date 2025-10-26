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

Runtime Flags
- gguf_te_impl: 'cuda_fp8' | 'hf' | 'cpu'
- gguf_te_kernel_required: true|false (error hard se não houver kernel)
- WAN_TE_TILE: tile de Cout (default 256) para linear FP8.

Status
- P1 (Linear FP8): done (tile dequant → GEMM) — correta e com baixo pico de VRAM.
- P2 (Attention FP8): placeholder (SDPA). Próximo passo: kernel com softmax estável.
- P3 (Encoder wiring): pendente — rota 'cuda_fp8' ainda em “scaffolded”.

