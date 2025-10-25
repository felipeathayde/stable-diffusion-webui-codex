WAN22 assets resolution (Diffusers + GGUF)

Summary
- Complementary WAN2.2 assets (tokenizer, text encoders, scheduler configs, optional VAE) are now mirrored under `apps/server/backend/huggingface/<org>/<repo>` and used from there instead of the model directory.
- Repo candidates are restricted to valid Diffusers repos (e.g. `Wan-AI/Wan2.2-I2V-A14B-Diffusers`). Ambiguous names like `Wan2.2-Image-to-Video-14B` are no longer attempted.
- GGUF text‑context fallback loads the Diffusers pipeline from the vendored mirror, avoiding incorrect lookups in the user model folder.

Where files are read from
- Primary model dir (user): provided `model_dir` or file parent; only used when it is a proper Diffusers repo.
- Vendored HF mirror: `apps/server/backend/huggingface/<org>/<repo>`; always preferred for tokenizer/encoders when the model dir is not a Diffusers checkout.

Environment controls
- `CODEX_WAN_DIFFUSERS_REPO`: override the repo id to mirror (e.g. `Wan-AI/Wan2.2-I2V-A14B-Diffusers`).
- `HF_TOKEN`: Hugging Face token for gated/private repos; used by both `huggingface_hub` and the HTTP fallback.

Notes
- Minimal files mirrored: configs, tokenizers, scheduler configs, and text‑encoder weights. VAE weights are not auto‑fetched; if you provide a VAE directory or file, it is used. Otherwise encode‑prompt proceeds without VAE.
- Logs clarify when the model dir is not a Diffusers repo and that the vendored mirror/GGUF fallback will be used.

