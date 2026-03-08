# apps/backend/runtime/pipeline_stages Overview
Date: 2025-10-30
Last Review: 2026-03-08
Status: Active

## Purpose
- Provide shared orchestration helpers for Codex generation pipelines (text, image, and video).
- Centralize stage logic (prompt normalization, sampler planning, init-image prep, video metadata) to keep use cases lightweight and consistent.

## Key Files
- `prompt_context.py` — Prompt parsing + prompt-derived controls (clip_skip, width/height overrides).
- `sampling_plan.py` — Scheduler normalization, noise settings, plan building, sampler/RNG preparation.
- `sampling_execute.py` — Sampler execution + live preview callback + latent dump diagnostics.
- `scripts.py` — Script hooks + extra network (LoRA) activation helpers.
- `image_io.py` — PIL/tensor conversions + optional hires decode helper.
- `hires_fix.py` — Hires-fix helpers (denoise→start_at_step mapping + init latents/conditioning prep via global upscalers).
- `tiling.py` — VAE tiling apply/restore toggles.
- `image_init.py` — Utilities for encoding img2img/img2vid init images into tensor+latent bundles.
- `masked_img2img.py` — Masked img2img (“inpaint”) helpers: mask normalize/invert/blur + full-res crop plan + latent mask enforcement inputs.
- `video.py` — Video plan builder, LoRA/sampler configuration, shared interpolation stage helpers, and metadata assembly.
- `__init__.py` — Package marker (intentionally no re-export facade; callers import modules directly).

## Notes
- Modules in this directory must stay dependency-light and only import from `apps.*` namespaces.
- Prefer adding new pipeline stages here rather than duplicating logic inside `apps/backend/use_cases/`.
- Helper functions should raise explicit errors; avoid silent fallbacks or catching broad exceptions.
- 2025-12-14: `build_video_plan()` reads `steps` + `guidance_scale` directly from the request; LoRA application uses a lazy import to keep the module dependency-light for non-LoRA users.
- 2025-12-14: Video plan defaults `steps` to 30 when an ad-hoc caller omits it (matching `/api/{txt2vid,img2vid}` defaults) to avoid drifting configs.
- 2026-01-01: `clip_skip` is now treated as a prompt control applied in `apply_prompt_context(...)` (before conditioning is computed); request-level `clip_skip` is merged into `PromptContext.controls` when no `<clip_skip:…>` tag is present.
- 2026-01-01: The live preview callback in `common.py` uses `runtime/live_preview.py` (method enum + decode helper) and records the sampling step in `backend_state` when emitting preview images.
- 2026-01-01: Preview-factor fitting/logging (least-squares latent→RGB `factors`/`bias`) lives in `runtime/live_preview.py` and can be enabled via `CODEX_DEBUG_PREVIEW_FACTORS=1` (used to derive `Approx cheap` mappings for new latent formats).
- 2026-01-02: Removed token merging application from `common.py`; `<merge:...>` / `<tm:...>` tags are stripped but have no effect.
- 2026-01-02: Added standardized file header docstrings to pipeline stage helper modules (doc-only change; part of rollout).
- 2026-01-03: Split the former `common.py` golema into focused modules and removed the stage re-export facade (`__init__.py` is now intentionally empty).
- 2026-01-06: `sampling_plan.py` scheduler normalization is strict (no alias/case normalization) and invalid schedulers now raise instead of falling back.
- 2026-01-08: `sampling_execute.py` passes `width/height` into `build_sampling_context(...)` so flow-match models with dynamic shifting (Flux) can resolve `flow_shift` from scheduler_config (fail-fast when missing).
- 2026-01-24: Live preview interval/method are applied per task via thread-local overrides (no process-global `os.environ` mutation); env vars remain as fallbacks only.
- 2026-01-25: `clip_skip=0` is now supported as an explicit “use default” sentinel (merged from request metadata or `<clip_skip:0>` prompt tags) and is passed through `apply_prompt_context(...)` to reset clip skip per job.
- 2026-01-27: `video.py:export_video(...)` now passes a task label into engine export so outputs land under `output/{txt2vid,img2vid}-videos/<date>/...` (stable dirs; UI serves via `/api/output/{rel_path}`).
- 2026-02-12: `video.py` now owns shared parsing/execution of `extras.video_interpolation` for all WAN video use-cases (txt2vid/img2vid/vid2vid) to keep Option A mode pipelines aligned.
- 2026-02-13: `video.py` now computes effective output FPS after interpolation (`resolve_video_output_fps`) and enforces fail-loud export semantics when `save_output=true` but the engine export does not return a saved artifact.
- 2026-02-21: `video.py::read_video_interpolation_options(...)` now parses `video_interpolation.enabled` via shared strict bool parsing and fails loud for non-string `model` or non-int/invalid `times` values (no permissive truthy coercion such as `bool(\"false\")`); `video.py::export_video(...)` also parses `save_output`/`saved` via strict bool parsing.
- 2026-02-21: `video.py` now preserves list-backed frame sequences through interpolation/result assembly instead of unconditional `list(...)` copies, reducing per-request frame duplication in WAN video pipelines.
- 2026-02-27: `video.py` now owns shared SeedVR2 upscaling stage parsing/execution (`read_video_upscaling_options`, `apply_video_upscaling`) and returns structured `video_upscaling` metadata for txt2vid/img2vid/vid2vid use-cases.
- 2026-03-01: `video.py` now exposes `prepare_base_snapshot_video_options(...)` so WAN video use-cases can persist a pre-post-process base video (`*_base` filename prefix) whenever `save_output=true` and upscaling/interpolation is enabled.
- 2026-02-01: Added `hires_fix.py` to centralize hires pass prep and fix `denoise` semantics (no more inverted `start_at_step` mapping).
- 2026-02-08: Sampling stages now carry typed ER-SDE options (`SamplingPlan.er_sde`) from plan build to execution; `execute_sampling(...)` forwards options into `CodexSampler.sample(...)` and latent diagnostics include effective ER-SDE metadata when sampler is `er sde`.
- 2026-02-09: `sampling_execute` now gates LoRA apply/reset on `codex_objects_after_applying_lora` capability and fails loud only when LoRA selections are present without engine support, while still resetting stale LoRA state for no-selection runs.
- 2026-02-25: `sampling_execute.py` now forwards denoiser-adjacent hook callbacks (`pre_denoiser_hook`, `post_denoiser_hook`) into `CodexSampler.sample(...)`; `masked_img2img.py::LatentMaskEnforcer` now exposes Forge-style pre/post denoiser blend hooks (noisy outside-mask preblend + init-latent outside-mask postblend), while retaining post-sample clamp behavior.
- 2026-02-25: `masked_img2img.py::LatentMaskEnforcer` now materializes masks/latents per runtime batch size (fail-loud on shape mismatch) and uses deterministic seed-derived base noise (no global RNG sampling inside denoiser hooks), fixing masked `per_step_clamp` multi-batch stability and seed reproducibility.
- 2026-02-25: `run_before_sampling_hooks(...)` no longer auto-applies extra-network LoRAs; sampling-path LoRA ownership is now single-source in `sampling_execute.execute_sampling(...)` to prevent duplicate apply in one sampling pass.
- 2026-02-18: `prompt_context.py` now merges request override `lora_path` entries into `PromptContext.loras` (dedup by path, prompt-tag LoRAs keep precedence/weight) so API-resolved LoRA SHA selections can flow into sampling without exposing SHA tokens in prompt/UI.
- 2026-02-18: `image_io.py::maybe_decode_for_hr(...)` now gates base decode by hires upscaler id (`latent:*` skips decode; pixel-space upscalers decode), replacing the stale `latent_scale_mode` sentinel that forced redundant decode in latent hires paths.
- 2026-02-18: `image_io.py::maybe_decode_for_hr(...)` no longer hardcasts decoded tensors to fp32; decode dtype now follows the engine VAE output so runtime compute/storage policy remains authoritative.
- 2026-02-21: `image_init.py::prepare_init_bundle(...)` now resizes non-masked img2img init images to `processing.width/height` (when provided) before VAE encode, preventing oversized source images from forcing full-resolution encode memory usage when output dims are smaller.
- 2026-02-21: `masked_img2img.py::prepare_masked_img2img_bundle(...)` now normalizes init-image + mask dimensions to `processing.width/height` before latent encode, so inpaint/full-res paths no longer require pre-matched source dimensions and avoid oversized encode memory spikes.
- 2026-03-06: `hires_fix.py` now dispatches hires prep by `sd_model.engine_id` (`sd*` via SD backend; `flux1`/`flux1_fill`/`flux1_chroma`/`zimage`/`anima` via flow-style latent prep; `flux1_kontext` via flow-style prep with `image_latents` continuation mode; `flux2` via the same upscale/crop prep, still returning `image_latents` continuation data for the dedicated FLUX.2 second pass). `HiresPreparation` keeps the typed shared contract (`latents`, optional `image_conditioning`, `continuation_mode`); unknown engines fail loud (no SD fallback).
- 2026-03-02: `sampling_execute.execute_sampling(...)` now exposes explicit `allow_txt2img_conditioning_fallback` control (default `True`), so callers that intentionally clear `image_conditioning` (Kontext/image-latents continuation) can disable implicit txt2img `c_concat` injection without changing default behavior for legacy callsites.
- 2026-03-06: `sampling_plan.py::apply_sampling_overrides(...)` no longer auto-resets the scheduler to a sampler default when prompt controls change only the sampler; it preserves the current scheduler and validates the resulting sampler/scheduler pair fail-loud instead.
- 2026-03-08: `sampling_plan.py::resolve_sampler_scheduler_override(...)` rejects legacy `use same*` sentinel strings (`use same sampler`, `use same scheduler`, `use same`); inheritance is represented only by omission or empty overrides.
- 2026-03-08: `video.py::configure_sampler(...)` now fails loud for `sampler='uni-pc bh2'` before bridge application, so video scheduler configuration reports unsupported bridge capabilities explicitly.
