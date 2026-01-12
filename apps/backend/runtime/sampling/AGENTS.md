# apps/backend/runtime/sampling Overview
<!-- tags: runtime, sampling, sigma, scheduler -->
Owner: Runtime Maintainers
Last Review: 2026-01-08
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
  - `CodexSampler` builds the sampling context, validates scheduler compatibility, and by default runs the **native** sampler loop (no k-diffusion dependency).
  - K-diffusion integration is not exposed via env vars (settings are Web UI / payload-driven). If k-diffusion routing is needed, port it behind an explicit Web UI setting instead of env toggles.
  - Restart sampler / UniPC helpers under `k_diffusion_extra` remain optional/experimental and must not be enabled via env flags.
- `__init__.py`
- Sampling inner loop no longer emits legacy low-VRAM print warnings; memory pressure handling remains via `memory_management.manager.get_free_memory` without stdout noise.

## Design Notes
- Scheduler vs Sampler: the scheduler here determines only the sigma sequence; the sampler integrator (Euler, DPM++ 2M, UniPC, etc.) is selected separately by `SamplerKind` in `driver.py`.
- Canonical names: sampler/scheduler strings are strict and must match `SamplerKind` / `SchedulerName` values (no alias mapping; empty values are invalid).
- Karras Sigmas: Euler in diffusers typically uses Karras sigmas when enabled. We intentionally reuse the Karras schedule for `EULER_DISCRETE`. The integrator determines ODE vs ancestral behavior.
- Simple schedule: `SIMPLE` is predictor-aware and always appends a terminal 0.
- Flow-match shift source-of-truth: for flow-match predictors (`prediction_type='const'`), the schedule uses `flow_shift` resolved from diffusers `scheduler_config.json` (vendored HF mirror or diffusers repo dir):
  - Fixed shift: applies `shift * t / (1 + (shift - 1) * t)` on a base `linspace(1→1/N)` ladder.
  - Dynamic shift (Flux): computes `mu(seq_len)` from config and uses `exp(mu)` (exponential) or `mu` (linear) as the effective shift; requires explicit `width/height`.
  - Z Image Turbo parity: when the predictor is `FlowMatchEulerPrediction(pseudo_timestep_range=1000)` we mirror diffusers `ZImagePipeline`: force `sigma_min=0.0`, use base `linspace(1→0)` and append terminal 0 (double-zero tail). Default `steps=9` yields ~8 effective updates (last `dt=0`).
- Precision guardrails: `driver.py` observes denoiser outputs for NaNs and escalates precision via `memory_management.manager.report_precision_failure` (bf16→fp16). Exhaustion raises with guidance to force fp32 manually.
- Diagnostics: set `CODEX_LOG_SAMPLER=1` to log sampler setup and per-step norms; set `CODEX_LOG_SIGMAS=1` to dump the sigma ladder (first/last and a compact summary) for schedule comparisons.
- 2025-12-12: Added opt-in deep logs for flow debugging: `CODEX_ZIMAGE_DEBUG=1` / `CODEX_ZIMAGE_DEBUG_SAMPLING_INNER=1` prints CFG routing + cond/uncond norms for the first few inner-loop calls.
- 2025-12-29: `sampling/__init__.py` is now an import-light facade; torch-bound code moved to `inner_loop.py` and is only loaded by `driver.py` during sampling (keeps `sampling.catalog` usable by the API/UI without pulling torch at import time). `inner_loop.py` still imports `cleanup_cache` lazily to avoid loading `runtime.ops` until cleanup is needed.
- 2026-01-01: Native `DPM++ 2M` now uses the log-sigma-time DPM-Solver++(2M) update (k-diffusion parity); `DPM++ 2M SDE` uses the midpoint log-sigma form with independent noise in the native path (no BrownianTree noise sampler).
- Distilled/turbo CFG: when unconditional conditioning is omitted (`uncond=None`), `sampling_function_inner` bypasses CFG interpolation and returns the conditional prediction directly. This allows Turbo models to run with `guidance_scale=0` without collapsing denoised to zeros.
- Flow precision: for flow-match predictors (`prediction_type='const'`, e.g. Flux/Z-Image), `driver.py` forces sampling latents to fp32 (matching diffusers schedulers) even if the core runs in bf16/fp16.

## Updates
- 2026-01-02: Added standardized file header docstrings to sampling facade helpers (`__init__.py`, `condition.py`, `registry.py`) (doc-only change; part of rollout).
- 2026-01-04: Sampling now treats the engine core as `codex_objects.denoiser`; ControlNet hooks are optional and only active when the denoiser patcher supports them.
- 2026-01-06: Removed sampler/scheduler alias maps; API/UI must send canonical names and invalid values now fail fast.
- 2026-01-06: Updated default schedulers for `pndm`/`uni-pc`/`ddpm` to match model_index-derived defaults (DDIM/simple/beta).
- 2026-01-08: Flow-match `flow_shift` is resolved from diffusers `scheduler_config.json` (fixed/dynamic) and injected into `SIMPLE` schedule construction; Flux dynamic shifting is resolution-aware and fails fast without `width/height`.
- 2026-01-08: Refreshed `registry.py` file header block to include the `_build_specs` helper in the Symbols list (doc-only change).

## Risks / Invariants
- `steps` must be `>= 1`; schedule always includes terminal sigma=0.
- The predictor provided by the model must expose `sigma_min`/`sigma_max` scalars; upstream code validates this.
- This module does not import from archived upstream snapshots and does not depend on external schedulers.
- `compile_conditions(cond)` invariants:
  - `cond=None` → `None` (semântica preservada).
  - `cond` tensor → tratado como cross-attn (B,S,C); erro se `ndim!=3`.
  - `cond` dict → deve conter `crossattn` (B,S,C) e `vector` (B,V); sem fallback.
  - Produz `model_conds` com chaves canônicas: `c_crossattn`, `y` e, quando aplicável, `c_concat`.

### Logging
- DEBUG log (logger `backend.runtime.sampling.condition`) registra shapes compilados.
- `CODEX_LOG_SAMPLER=1` logs sampler setup with scheduler name, prediction_type (from predictor/scheduler), sigma bounds, and the first few sigmas; per-step logs continue to show sigma transitions and latent norms.
- Sampler diagnostics (`CODEX_LOG_SAMPLER`) do not change sampler routing (native vs other backends).

### Pré-checagens antes do denoiser
- Após a montagem (`cond_cat(c)`), validação obrigatória:
  - `'c_crossattn'` existe e é Tensor 3D.
  - Se `model.diffusion_model.num_classes` não é `None`, exigir `'y'` (Tensor 2D).
- Alinhamento de `dtype/device` para `c_crossattn`, `y` e `c_concat` com o `input_x` antes do `apply_model`.

## Future Work (not yet ported)
- Additional schedule families (e.g., EDM/FlowMatch variants) can be added as new `SchedulerName` values with explicit behavior.
- Respect settings toggles like `use_old_karras_scheduler_sigmas` once UI flow is finalized.
