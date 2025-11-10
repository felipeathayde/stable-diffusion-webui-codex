# Troubleshooting — SDXL and WAN 2.2 Pipelines

This guide lists quick checks to validate text conditioning and runtime assembly for SDXL (image) and WAN 2.2 (video).

SDXL — Quick Checks
- Enable: `--debug-conditioning` to print SDXL conditioning stats.
- Expect:
  - `cond.crossattn` shape `(B, S, 2048)` and non‑zero norm for prompted runs.
  - `cond.vector` shape `(B, ~2816)` (1280 pooled + 1536 time/size ids).
- Errors you may see:
  - “UNet context feature dim mismatch”: CLIP concat not matching UNet `context_dim`.
  - “UNet ADM conditioning mismatch”: `y` missing while UNet expects ADM.
  - “UNet ADM feature mismatch”: `cond.vector` wrong length.
  - Fix: ensure SDXL model weights/unet config align with cross‑attn 2048 and ADM ~2816.

WAN 2.2 — Quick Checks (GGUF)
- Logs (set env): `WAN_LOG_INFO=1` (default), `WAN_LOG_DEBUG=1` for verbose.
- Expect:
  - Tokenizer line with dir, vocab size, `model_max_length`.
  - Tokenized shapes: batch=2 (prompt/negative), consistent seqlen.
  - TE outputs: prompt/negative tensors on the selected device/dtype.
  - Hidden size match: `last_hidden_state.shape[-1] == config.hidden_size`.
- CUDA FP8 path:
  - Requires `wan_te_cuda` built and available; otherwise explicit error.

References
- SDXL conditioning contract: `.sangoi/architecture/model-pipelines-bible.md`.
- Dev probe for SDXL (Diffusers): `tools/dev/sdxl_diffusers_probe.py`.

