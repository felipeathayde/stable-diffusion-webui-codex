# apps/backend/engines/zimage
Date: 2025-12-12
Owner: Engine Maintainers
Last Review: 2025-12-12
Status: Active

## Purpose
- Engine wiring for **Z Image Turbo** (ModelFamily `zimage`): loading core-only checkpoints, binding external Qwen3 text encoders + Flow16 VAEs, and exposing txt2img execution via the shared pipeline runner.

## Key Files
- `apps/backend/engines/zimage/spec.py` — Runtime assembly (external VAE/Qwen3 for core-only checkpoints) + flow predictor defaults.
- `apps/backend/engines/zimage/zimage.py` — `ZImageEngine` implementation (prompt formatting, conditioning, VAE encode/decode semantics).
- `apps/backend/engines/zimage/__init__.py` — Engine exports.

## References (vendored assets)
- `apps/backend/huggingface/Alibaba-TongYi/Z-Image-Turbo/scheduler/scheduler_config.json` — canonical `shift` + `num_train_timesteps`.
- `apps/backend/huggingface/Alibaba-TongYi/Z-Image-Turbo/vae/config.json` — canonical `scaling_factor` + `shift_factor`.

## Notes / Decisions
- **Distilled CFG:** Turbo models run single-branch conditioning (`use_distilled_cfg_scale=True`); negative prompts are ignored by design.
- **VAE normalization:** decode must apply `vae.first_stage_model.process_out(latents)` before `vae.decode(...)` (Flux/Z-Image latent format).

