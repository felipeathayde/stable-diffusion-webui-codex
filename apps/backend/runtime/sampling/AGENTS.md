# apps/backend/runtime/sampling Overview
<!-- tags: runtime, sampling, sigma, scheduler -->

## Purpose
- Native sampling primitives for Codex engines: sigma schedule construction and sampling context used by the runtime samplers.

## Key Components
- `context.py`
  - `SchedulerName` enum: canonical scheduler identifiers used to build sigma schedules (automatic/simple, karras/euler_discrete, exponential/polyexponential, uniform/sgm_uniform, ddim/ddim_uniform, normal, beta, linear_quadratic, kl_optimal, turbo, align-your-steps variants).
  - `build_sigma_schedule(...)`: returns a tensor of sigmas (length = steps + 1) ending with terminal 0; accepts a predictor to derive ladders for predictor-aware schedules (simple/uniform/normal/ddim/sgm_uniform/turbo) and an `is_sdxl` hint for AYS tables.
  - `SamplingContext`: immutable inputs for the sampler loop (sampler kind, schedule, noise settings, prediction type/sigma bounds).
- `registry.py`
  - `SamplerSpec` dataclass: canonical sampler name/kind, aliases, default scheduler, and allowed schedulers per sampler (Forge + ComfyUI surface).
  - `get_sampler_spec(name)`: resolves aliases and validates scheduler compatibility before sampling context creation.
- `driver.py`
  - `CodexSampler` builds the sampling context, validates scheduler compatibility, and by default runs the **native** sampler loop (no k-diffusion dependency).
  - K-diffusion-backed samplers are optional and only used when `CODEX_SAMPLER_ENABLE_KDIFFUSION=1` (and the `k-diffusion` Python package is installed); otherwise all sampling goes through the native path.
  - Restart sampler is provided via `k_diffusion_extra.restart_sampler`; UniPC/UniPC-BH2 are wired through `k_diffusion_extra` (BH2 currently reuses the UniPC update until a dedicated integrator is ported). These helpers require `k-diffusion` and are only reachable when k-diffusion is explicitly enabled.
- `__init__.py`
  - Sampling inner loop no longer emits Forge-era low-VRAM print warnings; memory pressure handling remains via `memory_management.get_free_memory` without stdout noise.

## Design Notes
- Scheduler vs Sampler: the scheduler here determines only the sigma sequence; the sampler integrator (Euler, DPM++ 2M, UniPC, etc.) is selected separately by `SamplerKind` in `driver.py`.
- Aliases: the API/UI may send diffusers class names. We normalize an explicit set (auto/simple, karras, exponential/polyexponential, euler_discrete, uniform/sgm_uniform, ddim/ddim_uniform, normal, beta, linear_quadratic, kl_optimal, turbo, align-your-steps variants). Unknown names raise `ValueError` with the supported list.
- Karras Sigmas: Euler in diffusers typically uses Karras sigmas when enabled. We intentionally reuse the Karras schedule for `EULER_DISCRETE`. The integrator determines ODE vs ancestral behavior.
- Simple schedule: `SIMPLE` is predictor-aware and always appends a terminal 0.
  - Default path mirrors Forge's `simple_scheduler`, sampling directly from the predictor's `sigmas` ladder (highest-to-lowest).
  - Z Image Turbo parity: when the predictor is `FlowMatchEulerPrediction(pseudo_timestep_range=1000)` we mirror diffusers `ZImagePipeline` behavior: force `sigma_min=0.0`, use a base `linspace(1→0)` ladder, apply `shift` once, and append terminal 0 (double-zero tail). Default `steps=9` yields ~8 effective updates (last `dt=0`).
- Precision guardrails: `driver.py` observes UNet outputs for NaNs and escalates precision via `memory_management.report_precision_failure` (bf16→fp16). Exhaustion raises with guidance to force fp32 manually.
- Diagnostics: set `CODEX_LOG_SAMPLER=1` to log sampler setup and per-step norms; set `CODEX_LOG_SIGMAS=1` to dump the sigma ladder (first/last and a compact summary) for schedule comparisons.
- 2025-12-12: Added opt-in deep logs for flow debugging: `CODEX_ZIMAGE_DEBUG=1` / `CODEX_ZIMAGE_DEBUG_SAMPLING_INNER=1` prints CFG routing + cond/uncond norms for the first few inner-loop calls.
- 2025-12-29: `sampling/__init__.py` now imports `cleanup_cache` lazily (prevents `apps.backend.runtime.sampling.catalog` consumers from pulling `runtime.ops` / torch-heavy modules at import time).
- Distilled/turbo CFG: when unconditional conditioning is omitted (`uncond=None`), `sampling_function_inner` bypasses CFG interpolation and returns the conditional prediction directly. This allows Turbo models to run with `guidance_scale=0` without collapsing denoised to zeros.
- Flow precision: for flow-match predictors (`prediction_type='const'`, e.g. Flux/Z-Image), `driver.py` forces sampling latents to fp32 (matching diffusers schedulers) even if the core runs in bf16/fp16.

## Risks / Invariants
- `steps` must be `>= 1`; schedule always includes terminal sigma=0.
- The predictor provided by the model must expose `sigma_min`/`sigma_max` scalars; upstream code validates this.
- This module does not import from `.refs/Forge-A1111/`, `.refs/InvokeAI/`, or `.refs/ComfyUI/` and does not depend on external schedulers.
- `compile_conditions(cond)` invariants:
  - `cond=None` → `None` (semântica preservada).
  - `cond` tensor → tratado como cross-attn (B,S,C); erro se `ndim!=3`.
  - `cond` dict → deve conter `crossattn` (B,S,C) e `vector` (B,V); sem fallback.
  - Produz `model_conds` com chaves canônicas: `c_crossattn`, `y` e, quando aplicável, `c_concat`.

### Logging
- DEBUG log (logger `backend.runtime.sampling.condition`) registra shapes compilados.
- `CODEX_LOG_SAMPLER=1` logs sampler setup with scheduler name, prediction_type (from predictor/scheduler), sigma bounds, and the first few sigmas; per-step logs continue to show sigma transitions and latent norms.
- K-diffusion routing is controlled exclusively via `CODEX_SAMPLER_ENABLE_KDIFFUSION=1`; enabling sampler logging alone (`CODEX_LOG_SAMPLER`) does not change whether k-diffusion is used.

### Pré-checagens antes do UNet
- Após a montagem (`cond_cat(c)`), validação obrigatória:
  - `'c_crossattn'` existe e é Tensor 3D.
  - Se `model.diffusion_model.num_classes` não é `None`, exigir `'y'` (Tensor 2D).
- Alinhamento de `dtype/device` para `c_crossattn`, `y` e `c_concat` com o `input_x` antes do `apply_model`.

## Future Work (not yet ported)
- Additional schedule families (e.g., EDM/FlowMatch variants) can be added as new `SchedulerName` values with explicit behavior.
- Respect settings toggles like `use_old_karras_scheduler_sigmas` once UI flow is finalized.
