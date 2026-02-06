# apps/backend/engines/anima Overview
<!-- tags: backend, engines, anima, cosmos -->
Date: 2026-02-05
Last Review: 2026-02-06
Status: Draft

## Purpose
- Host the Anima engine implementation (Cosmos Predict2 + Anima adapter) for txt2img/img2img.

## Key Files
- `apps/backend/engines/anima/anima.py` — `AnimaEngine` implementation (engine facade; delegates mode pipelines to use-cases per Option A).
- `apps/backend/engines/anima/spec.py` — Runtime spec + assembly (`assemble_anima_runtime`), including predictor config.
- `apps/backend/engines/anima/factory.py` — Factory that builds `CodexObjects` from the assembled runtime.

## Notes
- Anima uses sha-selected external assets (VAE + Qwen3-0.6B text encoder); no raw client paths.
- Engine assembly contract is explicit and fail-loud: `engine_options` must provide existing non-empty string file paths for `vae_path` + `tenc_path`.
- Capability exposure follows the Anima conditioning contract (`crossattn` + pooled `vector` + `t5xxl_ids/t5xxl_weights`); keep `AnimaEngine.capabilities()` in sync with `runtime/model_registry/capabilities.py`.
- Runtime device consistency checks normalize equivalent labels (`cuda` and `cuda:0`) before mismatch validation; only real device mismatches should fail, and missing `denoiser.model.load_device` fails loud.
- Conditioning requires dual tokenization (Qwen embeddings + T5 ids/weights), per `.sangoi/research/models/hf-circlestone-labs-anima.md`.
