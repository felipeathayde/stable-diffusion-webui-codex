<!-- tags: backend, huggingface, anima, tokenizers, offline -->
# apps/backend/huggingface/circlestone-labs/Anima Overview
Date: 2026-02-06
Last Review: 2026-02-06
Status: Active

## Purpose
- Vendored lightweight upstream docs/workflow and offline tokenizers required by the `anima` engine in strict/offline mode.
- This tree intentionally **does not** include Anima model weights (`*.safetensors`).

## Layout
- `LICENSE.md`, `README.md`, `anima_comparison.json`, `example.png` — upstream docs/workflow (downloaded via `hf`).
- `qwen25_tokenizer/` — Qwen tokenizer files used for Anima Qwen3-0.6B prompt tokenization.
- `t5_tokenizer/` — T5 tokenizer files used to produce Anima `t5xxl_ids`/`t5xxl_weights` (adapter ids/weights only; no T5 text encoder).

## Notes
- These tokenizers are copied from existing vendored HF mirrors under `apps/backend/huggingface/**` to keep Anima self-contained while matching ComfyUI’s `qwen25_tokenizer` + `t5_tokenizer` convention.
- Runtime code prefers these directories by default:
  - `apps/backend/runtime/families/anima/text_encoder.py`
