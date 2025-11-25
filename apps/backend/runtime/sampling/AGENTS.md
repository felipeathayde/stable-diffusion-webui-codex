# apps/backend/runtime/sampling Overview
<!-- tags: runtime, sampling, sigma, scheduler -->

## Purpose
- Native sampling primitives for Codex engines: sigma schedule construction and sampling context used by the runtime samplers.

## Key Components
- `context.py`
  - `SchedulerName` enum: canonical scheduler identifiers used to build sigma schedules.
  - `build_sigma_schedule(...)`: returns a tensor of sigmas (length = steps + 1) ending with terminal 0.
  - `SamplingContext`: immutable inputs for the sampler loop (sampler kind, schedule, noise settings, prediction type/sigma bounds).

## Design Notes
- Scheduler vs Sampler: the scheduler here determines only the sigma sequence; the sampler integrator (Euler, DPM++ 2M, UniPC, etc.) is selected separately by `SamplerKind` in `driver.py`.
- Aliases: the API/UI may send diffusers class names. We normalize a small, explicit set of aliases:
  - `EulerDiscreteScheduler`, `euler`, `euler a`, `EulerAncestralDiscreteScheduler` → `EULER_DISCRETE` schedule (Karras sigmas).
  - No silent fallbacks. Unknown names raise `ValueError` with the supported list.
- Karras Sigmas: Euler in diffusers typically uses Karras sigmas when enabled. We intentionally reuse the Karras schedule for `EULER_DISCRETE`. The integrator determines ODE vs ancestral behavior.
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
