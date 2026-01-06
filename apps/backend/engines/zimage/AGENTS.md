# apps/backend/engines/zimage
Date: 2025-12-12
Owner: Engine Maintainers
Last Review: 2026-01-06
Status: Active

## Purpose
- Engine wiring for **Z Image Turbo** (ModelFamily `zimage`): loading core-only checkpoints, binding external Qwen3 text encoders + Flow16 VAEs, and exposing txt2img execution via the shared pipeline runner.

## Key Files
- `apps/backend/engines/zimage/spec.py` — Runtime assembly (external VAE/Qwen3 for core-only checkpoints) + flow predictor defaults.
- `apps/backend/engines/zimage/factory.py` — Factory seam returning `(runtime, CodexObjects)` for consistent engine assembly.
- `apps/backend/engines/zimage/zimage.py` — `ZImageEngine` implementation (prompt formatting, conditioning, VAE encode/decode semantics).
- `apps/backend/engines/zimage/__init__.py` — Engine exports.

## References (vendored assets)
- `apps/backend/huggingface/Alibaba-TongYi/Z-Image-Turbo/scheduler/scheduler_config.json` — canonical `shift` + `num_train_timesteps`.
- `apps/backend/huggingface/Alibaba-TongYi/Z-Image-Turbo/vae/config.json` — canonical `scaling_factor` + `shift_factor`.

## Notes / Decisions
- **Distilled CFG:** Turbo models run single-branch conditioning (`use_distilled_cfg_scale=True`); negative prompts are ignored by design.
- **VAE normalization:** decode must apply `vae.first_stage_model.process_out(latents)` before `vae.decode(...)` (Flux/Z-Image latent format).
- **Prompt wrappers:** `ZImageEngine._prepare_prompt_wrappers(...)` attaches `distilled_cfg_scale` to the prompt list from `processing.distilled_guidance_scale` so the UI “guidance” slider can be propagated into conditioning logs (and any future guidance embedding usage).
- **Diffusers-math sampler:** `standalone_sampler.sample_zimage_diffusers_math(...)` mirrors diffusers scheduler behavior (`shift=3.0`, `sigma_min=0.0`) and avoids double-negating the model output (core already returns `noise_pred=-v`).
- **Debugging:** set `CODEX_ZIMAGE_DEBUG_PROMPT=1` to log the formatted prompt string and `distilled_cfg_scale` used for the run.
- 2026-01-01: ZImage prompt conditioning now participates in `smart_cache` (`zimage.conditioning`) so repeated prompts don’t re-encode Qwen3 each time; `get_learned_conditioning(...)` returns the cross-attn tensor directly (no placeholder `vector/guidance` allocations).
- 2026-01-02: Added standardized file header docstrings to Z Image engine modules (doc-only change; part of rollout).
- 2026-01-03: Z Image runtime core is now stored as `ZImageEngineRuntime.denoiser` via `DenoiserPatcher` (no ControlNet graph).
- 2026-01-03: `ZImageEngine` now assembles via `CodexZImageFactory` (factory-first seam; reduces drift in `_build_components`).
- 2026-01-06: Z Image uses `vae_path`/`tenc_path` as **external asset selection** (not state-dict overrides); core-only (`.gguf`) checkpoints require them, and full checkpoints may optionally override embedded assets by providing them.
