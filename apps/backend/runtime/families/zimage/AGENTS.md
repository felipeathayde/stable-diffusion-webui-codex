# apps/backend/runtime/families/zimage
Date: 2025-12-12
Owner: Runtime Maintainers
Last Review: 2026-01-30
Status: Active

## Purpose
- Codex-native runtime implementation for **Z Image** (Turbo/Base variants): the DiT core (`ZImageTransformer2DModel`), RoPE utilities, and Qwen3 text-encoder runtime used by the Z Image engine.

## Key Files
- `apps/backend/runtime/families/zimage/model.py` — Z Image DiT core + RoPE embedding (consumes HF `axes_dims`/`t_scale` config keys).
- `apps/backend/runtime/families/zimage/text_encoder.py` — Qwen3-4B text encoder wrapper + GGUF/state_dict loading.
- `apps/backend/runtime/families/zimage/inference.py` — Shared dimension inference (detector + loader) for prefixed exports and GGUF-core state dicts.
- `apps/backend/runtime/families/zimage/qwen3.py` — Native Qwen3 modules + attention-mask / SDPA debug helpers.
- `apps/backend/runtime/families/zimage/debug.py` — Opt-in debug helpers (env flags, tensor stats, token summaries).

## References (vendored assets)
- `apps/backend/huggingface/Tongyi-MAI/Z-Image-Turbo/transformer/config.json` — canonical `rope_theta`, `axes_dims`, `t_scale`, dims.
- `apps/backend/huggingface/Tongyi-MAI/Z-Image-Turbo/text_encoder/config.json` — canonical `hidden_size` (context dim).
  - Base mirror: `apps/backend/huggingface/Tongyi-MAI/Z-Image/**` (same component layout; scheduler shift differs).

## Notes / Decisions
- **Timesteps:** model receives `sigma∈[0,1]`, uses `t_inv = 1 - sigma`, and applies `t_scale` (default 1000.0) before timestep embedding (HF config parity).
- **RoPE:** `axes_dims` are full head-dim units and must sum to `head_dim` (HF config default `(32,48,48)` for `head_dim=128`).
- **Token order + pos_ids:** match diffusers `transformer_z_image.py`:
  - unified sequence is **image tokens first**, **caption tokens after**.
  - caption pos_ids use `create_coordinate_grid(size=(cap_len_padded,1,1), start=(1,0,0))`.
  - image pos_ids use `create_coordinate_grid(size=(1,H//p,W//p), start=(cap_len_padded+1,0,0))` and padding uses `(0,0,0)`.
  - pad tokens are set via `x_pad_token` / `cap_pad_token` (pads are part of the sequence; attention mask only matters for cross-item padding).
- **Non-modulated blocks parity:** when a block runs without adaLN modulation (context refiner, or any `t_emb=None` path), it must follow diffusers ordering:
  - `attn_out = attention(attention_norm1(x))`
  - `x = x + attention_norm2(attn_out)`
  - `x = x + ffn_norm2(feed_forward(ffn_norm1(x)))`
  Avoid applying `attention_norm2(attention_norm1(x))` as attention input (double-norm) or feeding `ffn_norm2(...)` into the MLP.
- **VAE normalization:** Flow16 (Flux/Z-Image) scaling/shift is applied outside the runtime core via `vae.first_stage_model.process_in/out`.
- **Tokenizer source of truth:** prefer the vendored HF tokenizer (no hub fetch):
  - Turbo: `apps/backend/huggingface/Tongyi-MAI/Z-Image-Turbo/tokenizer`
  - Base: `apps/backend/huggingface/Tongyi-MAI/Z-Image/tokenizer`
  The engine sets a per-run tokenizer hint based on `extras.zimage_variant`; override with `CODEX_ZIMAGE_TOKENIZER_PATH` when needed.
- 2025-12-29: ZImage tokenizer fallback paths are now anchored to `CODEX_ROOT` (required) so tokenizers resolve correctly when the process CWD is not the repo root.
- 2026-01-04: ZImage GGUF text encoder loads GGUF via `apps/backend/runtime/checkpoint/io.py:load_gguf_state_dict` so global runtime flags (e.g. `--gguf-exec=dequant_upfront`) apply consistently.
- 2026-01-04: Added `runtime/families/zimage/inference.py` and updated detector/loader to use it (prevents drift in hidden/context/latent/layer inference; supports prefixed SafeTensors exports).
- 2026-01-02: Added standardized file header docstrings to Z-Image runtime facade/debug modules (doc-only change; part of rollout).
- 2026-01-23: Centralized llama.cpp-style GGUF tensor-name remapping for Qwen3 in `apps/backend/runtime/state_dict/keymap_llama_gguf.py`; `qwen3.py:remap_gguf_keys` delegates and fails loud on unknown keys.
- 2026-01-30: Fixed Qwen3 causal+padding attention-mask construction (avoids `0 * -inf` NaNs) and removed always-on debug logs; deep diagnostics remain opt-in behind `CODEX_ZIMAGE_DEBUG_*`.
- **Debugging:** enable extra logs with env flags:
  - `CODEX_ZIMAGE_DEBUG_PROMPT=1` (engine prompt string + distilled cfg scale)
  - `CODEX_ZIMAGE_DEBUG_TENC_TEXT=1`, `CODEX_ZIMAGE_DEBUG_TENC_TOKENS=1`, `CODEX_ZIMAGE_DEBUG_TENC_DECODE=1`, `CODEX_ZIMAGE_DEBUG_TENC_RUN=1` (tokenization + embedding stats)
  - `CODEX_ZIMAGE_DEBUG_CONFIG=1`, `CODEX_ZIMAGE_DEBUG_VERBOSE=1`, `CODEX_ZIMAGE_DEBUG_LAYERS=1` (model config/kwargs/per-layer stats; gated by `CODEX_ZIMAGE_DEBUG_STEPS`)
