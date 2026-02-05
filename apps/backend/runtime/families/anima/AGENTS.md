# apps/backend/runtime/families/anima Overview
<!-- tags: backend, runtime, families, anima, cosmos, predict2 -->
Date: 2026-02-05
Last Review: 2026-02-05
Status: Draft

## Purpose
- Host the Codex-native runtime for Anima (Cosmos Predict2 / MiniTrainDiT) used by the `anima` engine.

## Key Files
- `apps/backend/runtime/families/anima/config.py` — Runtime config dataclasses + state-dict shape inference.
- `apps/backend/runtime/families/anima/dit.py` — Cosmos Predict2 DiT building blocks + `MiniTrainDiT`.
- `apps/backend/runtime/families/anima/llm_adapter.py` — Anima `LLMAdapter` implementation (dual-tokenization adapter; weights in core checkpoint).
- `apps/backend/runtime/families/anima/model.py` — `AnimaDiT` wrapper (MiniTrainDiT + adapter glue).
- `apps/backend/runtime/families/anima/loader.py` — Strict loader utilities (fail-loud missing/unexpected key diagnostics).

## Notes
- Image inference uses 4D latents (`B,C,H,W`) in Codex sampling; Cosmos Predict2 core expects 5D (`B,C,T,H,W`). The Anima runtime must treat images as `T=1` and preserve shape on output.
- Sampling semantics must match ComfyUI discrete flow: `shift=3.0`, `multiplier=1.0`, prediction type `const` (see `.sangoi/research/models/hf-circlestone-labs-anima.md`).
- `wan_vae.py` performs explicit header-key variant detection (`2.1` vs `2.2`) before loading weights; Anima v1 currently ports `2.1` only and must fail loud on `2.2`.
- Do not copy `.refs/**` code into `apps/**`; extract intent and re-implement cleanly.
