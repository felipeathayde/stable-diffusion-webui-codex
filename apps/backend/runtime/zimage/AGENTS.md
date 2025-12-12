# apps/backend/runtime/zimage
Date: 2025-12-12
Owner: Runtime Maintainers
Last Review: 2025-12-12
Status: Active

## Purpose
- Codex-native runtime implementation for **Alibaba-TongYi/Z-Image-Turbo**: the DiT core (`ZImageTransformer2DModel`), RoPE utilities, and Qwen3 text-encoder runtime used by the Z Image engine.

## Key Files
- `apps/backend/runtime/zimage/model.py` — Z Image Turbo DiT core + RoPE embedding (consumes HF `axes_dims`/`t_scale` config keys).
- `apps/backend/runtime/zimage/text_encoder.py` — Qwen3-4B text encoder wrapper + GGUF/state_dict loading.
- `apps/backend/runtime/zimage/qwen3.py` — Native Qwen3 modules + attention-mask / SDPA debug helpers.
- `apps/backend/runtime/zimage/debug.py` — Opt-in debug helpers (env flags, tensor stats, token summaries).

## References (vendored assets)
- `apps/backend/huggingface/Alibaba-TongYi/Z-Image-Turbo/transformer/config.json` — canonical `rope_theta`, `axes_dims`, `t_scale`, dims.
- `apps/backend/huggingface/Alibaba-TongYi/Z-Image-Turbo/text_encoder/config.json` — canonical `hidden_size` (context dim).

## Notes / Decisions
- **Timesteps:** model receives `sigma∈[0,1]`, uses `t_inv = 1 - sigma`, and applies `t_scale` (default 1000.0) before timestep embedding (HF config parity).
- **RoPE:** `axes_dims` are full head-dim units and must sum to `head_dim` (HF config default `(32,48,48)` for `head_dim=128`).
- **VAE normalization:** Flow16 (Flux/Z-Image) scaling/shift is applied outside the runtime core via `vae.first_stage_model.process_in/out`.
- **Debugging:** enable extra logs with env flags:
  - `CODEX_ZIMAGE_DEBUG_PROMPT=1` (engine prompt string + distilled cfg scale)
  - `CODEX_ZIMAGE_DEBUG_TENC_TEXT=1`, `CODEX_ZIMAGE_DEBUG_TENC_TOKENS=1`, `CODEX_ZIMAGE_DEBUG_TENC_DECODE=1`, `CODEX_ZIMAGE_DEBUG_TENC_RUN=1` (tokenization + embedding stats)
  - `CODEX_ZIMAGE_DEBUG_CONFIG=1`, `CODEX_ZIMAGE_DEBUG_VERBOSE=1`, `CODEX_ZIMAGE_DEBUG_LAYERS=1` (model config/kwargs/per-layer stats; gated by `CODEX_ZIMAGE_DEBUG_STEPS`)
