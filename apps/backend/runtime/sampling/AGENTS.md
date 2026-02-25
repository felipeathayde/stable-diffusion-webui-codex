# apps/backend/runtime/sampling Overview
<!-- tags: runtime, sampling, sigma, scheduler -->
Last Review: 2026-02-25
Status: Active

## Purpose
- Native sampling primitives for Codex engines: sigma schedule construction and sampling context used by the runtime samplers.

## Key Components
- `sigma_schedules.py`
  - `SchedulerName` enum: canonical scheduler identifiers used to build sigma schedules (simple, karras/euler_discrete, exponential/polyexponential, uniform/sgm_uniform, ddim/ddim_uniform, normal, beta, linear_quadratic, kl_optimal, turbo, align-your-steps variants).
  - `build_sigma_schedule(...)`: returns a tensor of sigmas (length = steps + 1) ending with terminal 0; accepts a predictor to derive ladders for predictor-aware schedules (simple/uniform/normal/ddim/sgm_uniform/turbo) and an `is_sdxl` hint for AYS tables.
- `flow_shift_resolver.py`
  - `resolve_flow_shift_for_sampling(...)`: resolves the effective flow shift for flow-match predictors from diffusers `scheduler_config.json` (diffusers repo dir or vendored HF mirror).
- `context.py`
  - `SamplingContext`: immutable inputs for the sampler loop (sampler kind, schedule, noise settings, prediction type/sigma bounds).
  - `build_sampling_context(...)`: assembles per-run sampling state (sigma bounds, flow shift, noise settings) and delegates schedule building to `sigma_schedules.build_sigma_schedule`.
- `registry.py`
  - `SamplerSpec` dataclass: canonical sampler name/kind, default scheduler, and allowed schedulers per sampler (UI surface).
  - `get_sampler_spec(name)`: resolves sampler name and validates scheduler compatibility before sampling context creation.
- `driver.py`
  - `CodexSampler` builds the sampling context, validates scheduler compatibility, and runs the **native** sampler loop (no external sampler deps).
  - Supports optional latent hooks (`pre_denoiser_hook`, `post_denoiser_hook`, `post_step_hook`, `post_sample_hook`) for use-case-controlled postprocessing (e.g. Forge-style masked img2img enforcement).
  - Restart sampler / UniPC helpers under `sampling_adapters/extra.py` are optional/experimental and are not wired into the default driver.
- `__init__.py` — import-light public surface re-exporting the sampler/scheduler catalog (no torch-bound exports).
- Sampling inner loop no longer emits legacy low-VRAM print warnings; memory pressure handling remains via `memory_management.manager.get_free_memory` without stdout noise.

## Design Notes
- Scheduler vs Sampler: the scheduler here determines only the sigma sequence; the sampler integrator (Euler, DPM++ 2M, UniPC, etc.) is selected separately by `SamplerKind` in `driver.py`.
- Canonical names: sampler/scheduler strings are strict and must match `SamplerKind` / `SchedulerName` values (no alias mapping; empty values are invalid).
- Karras Sigmas: Euler in diffusers typically uses Karras sigmas when enabled. We intentionally reuse the Karras schedule for `EULER_DISCRETE`. The integrator determines ODE vs ancestral behavior.
- Simple schedule: `SIMPLE` is predictor-aware and always appends a terminal 0.
- 2026-02-08: `SIMPLE` now supports predictor-selected modes:
  - `flowmatch_shifted_linspace` (legacy/default for flow predictors),
  - `comfy_downsample_sigmas` (ComfyUI parity: downsample predictor sigma ladder from the tail, then append terminal 0).
  Anima explicitly uses the Comfy mode via its predictor spec; FlowMatch families remain on legacy mode unless opted in.
- Sigma dtype: the sigma ladder is always built and used in **fp32** (even when the diffusion core runs bf16/fp16). Casting sigmas to low precision quantizes the schedule/timestep mapping and can cause severe “golesma/checkerboard/noise soup” regressions (notably SDXL).
- Flow-match shift source-of-truth: for flow-match predictors (`prediction_type='const'`), the schedule uses `flow_shift` resolved from diffusers `scheduler_config.json` (vendored HF mirror or diffusers repo dir):
  - Fixed shift: applies `shift * t / (1 + (shift - 1) * t)` on a base `linspace(1→1/N)` ladder.
  - Dynamic shift (Flux): computes `mu(seq_len)` from config and uses `exp(mu)` (exponential) or `mu` (linear) as the effective shift; requires explicit `width/height`.
  - Z Image parity (Turbo/Base): flow shift comes from the variant’s scheduler config (Turbo `shift=3.0`, Base `shift=6.0`). For core-only checkpoints, the variant must be explicit (no guessing).
    When the predictor is `FlowMatchEulerPrediction(pseudo_timestep_range=1000)` we mirror diffusers `ZImagePipeline`: force `sigma_min=0.0`, use base `linspace(1→0)` and append terminal 0 (double-zero tail). Default `steps=9` yields ~8 effective updates (last `dt=0`).
- Precision guardrails: `driver.py` observes denoiser outputs for NaNs and escalates precision via `memory_management.manager.report_precision_failure` (bf16→fp16). Exhaustion raises with guidance to force fp32 manually.
- Diagnostics: set `CODEX_LOG_SAMPLER=1` to log sampler setup and per-step norms; set `CODEX_LOG_SIGMAS=1` to dump the sigma ladder (first/last and a compact summary) for schedule comparisons.
- CFG batching: set `CODEX_CFG_BATCH_MODE=fused|split` to control whether the inner loop attempts a fused cond+uncond forward. `fused` is best-effort and may fall back to `split` if it OOMs.
- Forced fused retry against memory heuristic is now opt-in only via `CODEX_CFG_FUSED_FORCE_RETRY=1`; default `0` keeps heuristic-selected split path (avoids deliberate overcommit/OOM probe by default).
- Profiling (debug): set `CODEX_PROFILE=1` to enable the global torch-profiler wrapper (sampling driver emits per-step `record_function` ranges and exports a Perfetto trace + summary under `logs/profiler/`).
- 2025-12-12: Added opt-in deep logs for flow debugging: `CODEX_ZIMAGE_DEBUG=1` / `CODEX_ZIMAGE_DEBUG_SAMPLING_INNER=1` prints CFG routing + cond/uncond norms for the first few inner-loop calls.
- 2026-01-18: `sampling/__init__.py` re-exports only the import-light sampler/scheduler catalog; torch-bound sampling internals remain in `inner_loop.py` / `driver.py`.
- 2026-01-01: Native `DPM++ 2M` uses the log-sigma-time DPM-Solver++(2M) update; `DPM++ 2M SDE` uses the midpoint log-sigma form with independent noise in the native path (no BrownianTree noise sampler).
- Distilled/turbo CFG: when unconditional conditioning is omitted (`uncond=None`), `sampling_function_inner` bypasses CFG interpolation and returns the conditional prediction directly. This allows Turbo models to run with `guidance_scale=0` without collapsing denoised to zeros.
- Flow precision: for flow-match predictors (`prediction_type='const'`, e.g. Flux/Z-Image), the sigma ladder remains **fp32** (schedule stability), but sampling latents now follow the core role dtype (same policy as SD/SDXL). If quality regresses, force core precision to fp32 explicitly instead of relying on implicit overrides.

## Updates
- 2026-01-02: Added standardized file header docstrings to sampling facade helpers (`__init__.py`, `condition.py`, `registry.py`) (doc-only change; part of rollout).
- 2026-01-04: Sampling now treats the engine core as `codex_objects.denoiser`; ControlNet hooks are optional and only active when the denoiser patcher supports them.
- 2026-01-06: Removed sampler/scheduler alias maps; API/UI must send canonical names and invalid values now fail fast.
- 2026-01-06: Updated default schedulers for `pndm`/`uni-pc`/`ddpm` to match model_index-derived defaults (DDIM/simple/beta).
- 2026-01-08: Flow-match `flow_shift` is resolved from diffusers `scheduler_config.json` (fixed/dynamic) and injected into `SIMPLE` schedule construction; Flux dynamic shifting is resolution-aware and fails fast without `width/height`.
- 2026-01-08: Refreshed `registry.py` file header block to include the `_build_specs` helper in the Symbols list (doc-only change).
- 2026-01-14: Flow-match runs now log which `scheduler_config.json` path is used to resolve `flow_shift` (diffusers repo dir vs vendored HF mirror) before building sigmas; sigma schedule logs now print `predict_min/max` correctly when the bound is 0.
- 2026-01-24: Live preview interval is now resolved via thread-local overrides first (no per-task `os.environ` mutation); env `CODEX_PREVIEW_INTERVAL` remains as a fallback/debug input.
- 2026-01-26: Sampling driver enforces smart-offload pre-sampling residency (TE must be off; VAE allowed only for live preview `FULL` with a warning about VRAM/perf).
- 2026-01-30: Flow-match sampling no longer forces latents to fp32; latents now follow the core role dtype (SDXL parity). Sigma ladders remain fp32.
- 2026-01-31: Added opt-in global profiling (`CODEX_PROFILE`) at sampling seams to attribute time to per-step regions, model calls, and CPU↔GPU transfers.
- 2026-02-07: `ConditionCrossAttn` now fails loud on non-tensor/invalid-rank/zero-length cross-attn inputs in `can_concat` and `concat` to prevent raw divide-by-zero failures during concat math.
- 2026-02-08: Added Comfy SIMPLE parity branch for discrete flow predictors (`simple_schedule_mode="comfy_downsample_sigmas"`) with strict fail-loud guards (`steps>=1`, 1D finite monotone `predictor.sigmas`) and kept FlowMatch SIMPLE path unchanged by default.
- 2026-02-08: Applied low-risk inner-loop vectorization pass (P1/P2/P4): tensorized `compute_cond_mark`/`compute_cond_indices`, reduced Python-side batch assembly overhead, replaced repeated sigma cat with repeat-shape equivalent, and vectorized edge feathering with explicit tiny-area legacy fallback to preserve behavior.
- 2026-02-08: `driver.py` now includes native `SamplerKind.ER_SDE` execution with strict option normalization (`solver_type`, `max_stage`, `eta`, `s_noise`), finite/positivity guards on ER-SDE stage math, and fail-loud runtime errors for invalid integration states.
- 2026-02-16: `inner_loop.sampling_prepare(...)` is now self-cleaning on failure: if post-load setup fails (e.g., GGUF dequant cache/config/control-prepare), it immediately calls `sampling_cleanup(...)`; if cleanup also fails, the function raises a combined fail-loud error with both prepare and cleanup causes.
- 2026-02-18: `driver.py`/`inner_loop.py` now support optional guidance policy wiring (env + `override_settings.guidance`) for APG, CFG truncation by progress ratio, guidance rescale, and renorm clamp; policy parsing is strict and inactive policies are ignored (legacy sampling preserved when unset).
- 2026-02-18: `inner_loop.sampling_prepare(...)` / `sampling_cleanup(...)` now route smart-offload load/unload context through `memory_management.manager` (`source`/`stage`), keeping generic action emission (`load`/`unload`) centralized in the manager.
- 2026-02-20: `driver.py` now emits dense sampler diagnostics through `emit_backend_event(...)` (`sampling.sigma_schedule`, `sampling.plan.prepare`, `sampling.plan.run`, `sampling.cfg_delta`, `sampling.step`, `guidance.policy`) so logs inherit centralized multiline formatting/colorization instead of ad-hoc single-line format strings.
- 2026-02-22: Removed run-scoped GGUF dequant-forward cache hooks (`lvl1`/`lvl2`) from `inner_loop.sampling_prepare(...)`/`sampling_cleanup(...)`; sampling lifecycle now no longer enables/disables per-run GGUF forward caches.
- 2026-02-25: `driver.py` now applies optional denoiser-adjacent hooks per step (`pre_denoiser_hook` before denoise, `post_denoiser_hook` after denoise), enabling Forge-style masked blending semantics in img2img while preserving existing post-step and post-sample hook contracts.

## Risks / Invariants
- `steps` must be `>= 1`; schedule always includes terminal sigma=0.
- The predictor provided by the model must expose `sigma_min`/`sigma_max` scalars; upstream code validates this.
- This module does not import from archived upstream snapshots and does not depend on external schedulers.
- `compile_conditions(cond)` invariants:
  - `cond=None` → `None` (semântica preservada).
  - `cond` tensor → tratado como cross-attn (B,S,C); erro se `ndim!=3`.
  - `cond` dict → deve conter `crossattn` (B,S,C) e `vector` (B,V); sem fallback.
  - `t5xxl_ids`/`t5xxl_weights` são opcionais, mas obrigatoriamente em par; ambos 2D (B,S), com batch alinhado a `crossattn`.
  - Produz `model_conds` com chaves canônicas: `c_crossattn`, `y` e, quando aplicável, `c_concat`/`guidance`/`image_latents`/`t5xxl_ids`/`t5xxl_weights`.

### Logging
- DEBUG log (logger `backend.runtime.sampling.condition`) registra shapes compilados.
- `CODEX_LOG_SAMPLER=1` logs sampler setup with scheduler name, prediction_type (from predictor/scheduler), sigma bounds, and the first few sigmas; setup/step diagnostics now use structured `event + key=value` output (multiline for dense entries).
- Sampler diagnostics (`CODEX_LOG_SAMPLER`) do not change sampler routing (native vs other backends).

### Pré-checagens antes do denoiser
- Após a montagem (`cond_cat(c)`), validação obrigatória:
  - `'c_crossattn'` existe e é Tensor 3D.
  - Se `model.diffusion_model.num_classes` não é `None`, exigir `'y'` (Tensor 2D).
- Alinhamento de `dtype/device` para `c_crossattn`, `y` e `c_concat` com o `input_x` antes do `apply_model`.

## Future Work (not yet ported)
- Additional schedule families (e.g., EDM/FlowMatch variants) can be added as new `SchedulerName` values with explicit behavior.
- Respect settings toggles like `use_old_karras_scheduler_sigmas` once UI flow is finalized.
