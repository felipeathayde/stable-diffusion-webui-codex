# apps/backend/runtime/families/anima Overview
<!-- tags: backend, runtime, families, anima, cosmos, predict2 -->
Date: 2026-02-05
Last Review: 2026-02-23
Status: Draft

## Purpose
- Host the Codex-native runtime for Anima (Cosmos Predict2 / MiniTrainDiT) used by the `anima` engine.

## Key Files
- `apps/backend/runtime/families/anima/config.py` — Runtime config dataclasses + state-dict shape inference.
- `apps/backend/runtime/families/anima/dit.py` — Cosmos Predict2 DiT building blocks + `MiniTrainDiT`.
- `apps/backend/runtime/families/anima/llm_adapter.py` — Anima `LLMAdapter` implementation (dual-tokenization adapter; weights in core checkpoint).
- `apps/backend/runtime/families/anima/model.py` — `AnimaDiT` wrapper (MiniTrainDiT + adapter glue).
- `apps/backend/runtime/families/anima/text_encoder.py` — Qwen3-0.6B text encoder loader + offline tokenizers (Qwen + T5 ids/weights).
- `apps/backend/runtime/families/anima/loader.py` — Strict loader utilities (fail-loud missing/unexpected key diagnostics).
- `apps/backend/runtime/families/anima/wan_vae.py` — WAN 2.1 image-mode VAE (T=1) + strict header inference / variant gating.

## Notes
- Image inference uses 4D latents (`B,C,H,W`) in Codex sampling; Cosmos Predict2 core expects 5D (`B,C,T,H,W`). The Anima runtime must treat images as `T=1` and preserve shape on output.
- Sampling semantics must match ComfyUI discrete flow: `shift=3.0`, `multiplier=1.0`, prediction type `const` (see `.sangoi/research/models/hf-circlestone-labs-anima.md`).
- `text_encoder.py` resolves offline tokenizers from `apps/backend/huggingface/circlestone-labs/Anima/{qwen25_tokenizer,t5_tokenizer}` by default (override via `CODEX_ANIMA_QWEN_TOKENIZER_PATH` / `CODEX_ANIMA_T5_TOKENIZER_PATH`).
- `text_encoder.py` enforces non-empty Qwen token batches (`min_length=1` behavior): if empty prompts tokenize to `S=0`, it synthesizes one masked pad token and fails loud when `pad_token_id` metadata is invalid.
- 2026-02-07: `text_encoder.py` now exposes `tokenize_qwen_with_weights(..., return_word_ids=...)` plus adapter methods (`AnimaQwenTextEncoder.tokenize_with_weights`, `AnimaQwenTextProcessingEngine.tokenize_with_weights`). `word_id` is segment-based (per `parse_prompt_attention` segment), tuple handling is index-robust, and malformed metadata fails loud.
- `wan_vae.py` performs explicit header-key variant detection (`2.1` vs `2.2`) before weight load; Anima v1 currently ports `2.1` only and must fail loud on `2.2`.
- `wan_vae.py` infers `dim` from `decoder.head.0.gamma.shape` and expects broadcastable `(dim, 1, 1, 1)` (WAN 2.1); errors distinguish missing vs invalid shape, and `encoder.conv1.weight.shape[0]` must match `dim` (fail loud on mismatch).
- Do not copy `.refs/**` code into `apps/**`; extract intent and re-implement cleanly.
- 2026-02-08: `loader.py`, `wan_vae.py`, and `text_encoder.py` now run strict family keymaps from `apps.backend.runtime.state_dict.keymap_anima` before strict model load; parser `net.` stripping invariant for core remains enforced.
- 2026-02-08: `wan_vae.py` now populates Wan21 per-channel latent stats on `WanVaeConfig` (`latents_mean`/`latents_std`) for decode/encode normalization parity with Comfy; constructor enforces `z_dim=16` fail-loud for Anima image-mode scope.
- 2026-02-08: `wan_vae.py` now emits `shift_factor=None` (not numeric `0.0`) for Anima Wan21 parity; shared VAE no-shift policy remains strict and continues to reject explicit numeric shift values for Anima.
- 2026-02-16: WAN2.1 VAE keymap ownership moved to `apps/backend/runtime/state_dict/keymap_wan21_vae.py`; `wan_vae.py` now consumes model-owned keymap directly (no Anima mixed-ownership).
- 2026-02-20: Anima DiT and LLM adapter attention callsites now route through runtime dispatcher helper `attention_function_pre_shaped(...)` (explicit PyTorch backend override), removing direct SDPA bypasses in family modules.
- 2026-02-23: `text_encoder.py` device metadata fallback now resolves from memory-manager CPU device (`manager.cpu_device`) instead of constructing a local CPU literal when parameter iterators are empty.
- 2026-02-23: `dit.py` now mirrors Cosmos fp16 residual-stream safety intent: fp16 residual stream is promoted to fp32 before block loop, each block keeps module compute at embedding dtype (compute/residual split), and final layer input is cast back to context dtype.
- 2026-02-23: `llm_adapter.py` rotary tables are now generated under autocast-disabled fp32 math and `_apply_rotary(...)` performs fp32 rotary math for fp16/bf16 paths, then restores the original tensor dtype.
