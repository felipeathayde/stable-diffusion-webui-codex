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
  - `CodexSampler` builds the sampling context, validates scheduler compatibility, and by default dispatches *all* samplers through k-diffusion when available; set `CODEX_SAMPLER_FORCE_NATIVE=1` to force the legacy native loop.
  - Restart sampler is served via `k_diffusion_extra.restart_sampler`; UniPC/UniPC-BH2 are wired through `k_diffusion_extra` (BH2 currently reuses the UniPC update until a dedicated integrator is ported).
- `__init__.py`
  - Sampling inner loop no longer emits Forge-era low-VRAM print warnings; memory pressure handling remains via `memory_management.get_free_memory` without stdout noise.

## Design Notes
- Scheduler vs Sampler: the scheduler here determines only the sigma sequence; the sampler integrator (Euler, DPM++ 2M, UniPC, etc.) is selected separately by `SamplerKind` in `driver.py`.
- Aliases: the API/UI may send diffusers class names. We normalize an explicit set (auto/simple, karras, exponential/polyexponential, euler_discrete, uniform/sgm_uniform, ddim/ddim_uniform, normal, beta, linear_quadratic, kl_optimal, turbo, align-your-steps variants). Unknown names raise `ValueError` with the supported list.
- Karras Sigmas: Euler in diffusers typically uses Karras sigmas when enabled. We intentionally reuse the Karras schedule for `EULER_DISCRETE`. The integrator determines ODE vs ancestral behavior.
- Simple schedule: `SIMPLE` mirrors Forge's `simple_scheduler`, sampling directly from the predictor's `sigmas` ladder (highest-to-lowest) and appending a terminal 0. Euler/Euler a default to this schedule when the UI scheduler is "Automatic".
- Precision guardrails: `driver.py` observes UNet outputs for NaNs and escalates precision via `memory_management.report_precision_failure` (bf16→fp16). Exhaustion raises with guidance to force fp32 manually.
- Diagnostics: set `CODEX_LOG_SAMPLER=1` to log sampler setup and per-step norms; set `CODEX_LOG_SIGMAS=1` to dump the sigma ladder (first/last and a compact summary) for schedule comparisons.

## Risks / Invariants
- `steps` must be `>= 1`; schedule always includes terminal sigma=0.
- The predictor provided by the model must expose `sigma_min`/`sigma_max` scalars; upstream code validates this.
- This module does not import from `.legacy/` and does not depend on external schedulers.
- `compile_conditions(cond)` invariants:
  - `cond=None` → `None` (semântica preservada).
  - `cond` tensor → tratado como cross-attn (B,S,C); erro se `ndim!=3`.
  - `cond` dict → deve conter `crossattn` (B,S,C) e `vector` (B,V); sem fallback.
  - Produz `model_conds` com chaves canônicas: `c_crossattn`, `y` e, quando aplicável, `c_concat`.

### Logging
- DEBUG log (logger `backend.runtime.sampling.condition`) registra shapes compilados.
- `CODEX_LOG_SAMPLER=1` logs sampler setup with scheduler name, prediction_type (from predictor/scheduler), sigma bounds, and the first few sigmas; per-step logs continue to show sigma transitions and latent norms.

### Pré-checagens antes do UNet
- Após a montagem (`cond_cat(c)`), validação obrigatória:
  - `'c_crossattn'` existe e é Tensor 3D.
  - Se `model.diffusion_model.num_classes` não é `None`, exigir `'y'` (Tensor 2D).
- Alinhamento de `dtype/device` para `c_crossattn`, `y` e `c_concat` com o `input_x` antes do `apply_model`.

## Future Work (not yet ported)
- Additional schedule families (e.g., EDM/FlowMatch variants) can be added as new `SchedulerName` values with explicit behavior.
- Respect settings toggles like `use_old_karras_scheduler_sigmas` once UI flow is finalized.
