# apps/backend/engines/zimage
Date: 2025-12-12
Owner: Engine Maintainers
Last Review: 2026-01-30
Status: Active

## Purpose
- Engine wiring for **Z Image** (Turbo/Base variants; ModelFamily `zimage`): loading core-only checkpoints, binding external Qwen3 text encoders + Flow16 VAEs, and exposing txt2img/img2img execution via the shared pipeline runner.

## Key Files
- `apps/backend/engines/zimage/spec.py` — Runtime assembly (external VAE/Qwen3 for core-only checkpoints) + flow predictor defaults.
- `apps/backend/engines/zimage/factory.py` — Factory seam returning `(runtime, CodexObjects)` for consistent engine assembly.
- `apps/backend/engines/zimage/zimage.py` — `ZImageEngine` implementation (prompt formatting, conditioning, VAE encode/decode semantics).
- `apps/backend/engines/zimage/__init__.py` — Package marker (no re-exports); import from `spec.py` / `zimage.py`.

## References (vendored assets)
- `apps/backend/huggingface/Tongyi-MAI/Z-Image-Turbo/scheduler/scheduler_config.json` — Turbo canonical `shift` + `num_train_timesteps` (`shift=3.0`).
- `apps/backend/huggingface/Tongyi-MAI/Z-Image/scheduler/scheduler_config.json` — Base canonical `shift` + `num_train_timesteps` (`shift=6.0`).
- `apps/backend/huggingface/Tongyi-MAI/Z-Image-Turbo/vae/config.json` — canonical `scaling_factor` + `shift_factor`.

## Notes / Decisions
- **Variant contract:** UI sends `extras.zimage_variant="turbo"|"base"`; the backend forwards it to `engine_options["zimage_variant"]` so the orchestrator reloads the engine when the variant changes.
  - For Codex-produced GGUFs, the engine may also trust `codex.zimage.variant` metadata when it matches Codex provenance.
- **CFG semantics (diffusers parity):** Z-Image uses classic CFG; unconditional conditioning is used when `guidance_scale > 1` and negative prompts are supported for both variants.
- **VAE normalization:** decode must apply `vae.first_stage_model.process_out(latents)` before `vae.decode(...)` (Flux/Z-Image latent format).
- **Prompt wrappers:** `ZImageEngine._prepare_prompt_wrappers(...)` attaches `cfg_scale` to the prompt list from `processing.guidance_scale` so the UI “guidance” slider can be propagated into conditioning logs (and any future guidance embedding usage).
- **Diffusers-math sampler:** `standalone_sampler.sample_zimage_diffusers_math(...)` mirrors diffusers scheduler behavior (`shift=3.0`, `sigma_min=0.0`) and avoids double-negating the model output (core already returns `noise_pred=-v`).
- **Debugging:** set `CODEX_ZIMAGE_DEBUG_PROMPT=1` to log the formatted prompt string and `cfg_scale` used for the run.
- 2026-01-01: ZImage prompt conditioning now participates in `smart_cache` (`zimage.conditioning`) so repeated prompts don’t re-encode Qwen3 each time; `get_learned_conditioning(...)` returns the cross-attn tensor directly (no placeholder `vector/guidance` allocations).
- 2026-01-02: Added standardized file header docstrings to Z Image engine modules (doc-only change; part of rollout).
- 2026-01-03: Z Image runtime core is now stored as `ZImageEngineRuntime.denoiser` via `DenoiserPatcher` (no ControlNet graph).
- 2026-01-03: `ZImageEngine` now assembles via `CodexZImageFactory` (factory-first seam; reduces drift in `_build_components`).
- 2026-01-18: Z Image treats `vae_path`/`tenc_path` as **external asset selection** (not state-dict overrides) and the API requires sha-based selection (`vae_sha`/`tenc_sha`) for Z Image runs (no silent fallbacks).
- 2026-01-06: Refreshed `spec.py` header block wording to reflect optional external overrides for full checkpoints (doc-only change).
- 2026-01-08: `spec.flow_shift` now resolves from the vendored diffusers `scheduler_config.json` (HF mirror) instead of using family defaults, keeping scheduler parity as the source of truth.
- 2026-01-20: Removed unused dev-only ZImage artifacts (`diffusers_pipeline.py`, `test_diffusers.py`) — engine wiring lives in `spec.py` / `factory.py` / `zimage.py`.
- 2026-01-30: Removed dev-only Diffusers bypass flag (`CODEX_ZIMAGE_DIFFUSERS_BYPASS`) and downgraded Z-Image assembly/sampler logs to debug (default runs are quiet). External VAE/TEnc loading now follows memory-manager role dtypes (options `codex_vae_dtype` / `codex_te_dtype`) instead of a Z-Image-specific `dtype` override path.
- 2026-01-31: `ZImageEngine.encode_first_stage/decode_first_stage` now delegate to the shared `CodexDiffusionEngine` VAE stage implementation (timeline spans preserved; no behavior change).
