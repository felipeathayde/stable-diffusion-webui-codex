# apps/backend/runtime/sampling Overview
<!-- tags: runtime, sampling, sigma, scheduler -->
Last Review: 2026-03-08
Status: Active

## Purpose
- Native sampling primitives for Codex engines: sigma schedule construction, sampling context assembly, and native sampler execution.

## Key Components
- `sigma_schedules.py`
  - `SchedulerName` enum: canonical scheduler identifiers for sigma-ladder construction.
  - `build_sigma_schedule(...)`: builds sigma ladders for all exposed schedulers.
  - Predictor-backed schedulers (`simple`, `uniform`, `normal`, `beta`, `ddim`, `sgm_uniform`, `turbo`) require predictor data.
  - `beta` uses a predictor-ladder contract: Beta inverse-CDF over timestep probabilities, rounded ladder indices (duplicates preserved), exact requested non-terminal step count, and one terminal zero.
- `flow_shift_resolver.py`
  - `resolve_flow_shift_for_sampling(...)`: resolves effective flow shift for flow-match predictors from `scheduler_config.json` sources.
- `context.py`
  - `SamplingContext`: immutable per-run sampler/scheduler/noise setup.
  - `build_sampling_context(...)`: resolves sigma bounds + flow shift and builds the active sigma schedule.
- `deis.py`
  - `build_deis_coefficients(...)`: DEIS coefficient table builder for the native DEIS lane.
- `log_snr.py`
  - Shared half-logSNR helpers used by native stochastic logSNR samplers.
- `sa_solver.py`
  - SA-Solver-specific coefficient and tau helpers used by native `sa-solver` / `sa-solver pece` lanes.
- `registry.py`
  - `SamplerSpec` dataclass and `get_sampler_spec(name)` for canonical sampler/scheduler compatibility resolution.
- `driver.py`
  - `CodexSampler` native runtime driver.
  - Runs native lanes including `ddpm` with deterministic driver-owned `ImageRNG` stochastic draws.
  - Handles progress/cancellation/preview hooks under the driver contract.
- `__init__.py`
  - Import-light public surface for sampler/scheduler catalog exports.

## Current Contracts
- Scheduler vs sampler split is strict:
  - Scheduler controls only the sigma ladder.
  - Sampler integrator is selected by `SamplerKind` in `driver.py`.
- Canonical names are strict:
  - No alias mapping.
  - Empty or unknown sampler/scheduler names fail fast.
- Sigma ladder precision is fp32:
  - Schedules are built and consumed in fp32 to protect timestep/sigma mapping stability.
- `ddpm + beta` seam:
  - `ddpm` executes natively in `driver.py`.
  - `beta` is predictor-ladder based and no longer uses continuous sigma interpolation.
  - Effective denoise-step count matches the requested step count exactly; if the ladder cannot satisfy the count invariant, schedule construction fails loud.
- `dpm++ 2m cfg++` seam:
  - `dpm++ 2m cfg++` executes natively in `driver.py`.
  - The driver captures `uncond_denoised` through the sampler post-CFG hook and uses the dedicated CFG++ recurrence instead of aliasing plain `dpm++ 2m`.
  - Public scheduler exposure is intentionally narrowed to `karras` until broader parity is proven.
- `dpm++ 2m sde` seam:
  - Base `dpm++ 2m sde` executes natively in `driver.py` with prediction-type-aware half-logSNR recurrence.
  - Stochastic renoise is driver-owned and deterministic (`ImageRNG` seeded-step draws); ambient randomness is not used.
  - Partial-denoise resume burn is deterministic and consumes one draw per skipped positive interval (none for terminal `sigma_next == 0`).
  - Public scheduler exposure is intentionally narrowed to `exponential`.
- `euler cfg++` / `euler a cfg++` seam:
  - `euler cfg++` and `euler a cfg++` execute natively in `driver.py`.
  - Both lanes use the driver post-CFG hook to capture `uncond_denoised`; `euler a cfg++` also uses driver-owned deterministic `ImageRNG` noise.
  - Public scheduler exposure is intentionally narrowed to `euler_discrete` until broader parity is proven.
- `dpm++ 2s ancestral cfg++` seam:
  - `dpm++ 2s ancestral cfg++` executes natively in `driver.py`.
  - The lane captures `uncond_denoised` through the sampler post-CFG hook and uses a dedicated CFG++ ancestral recurrence instead of aliasing plain `dpm++ 2s ancestral`.
  - Stochastic renoise uses driver-owned deterministic `ImageRNG`; RF/CONST midpoint projection uses the shared half-logSNR helper seam.
  - Public scheduler exposure is intentionally narrowed to `karras` until broader parity is proven.
- Inventory-only sampler rows:
  - Raw `/api/samplers` inventory may still list unsupported rows with `supported=false`.
  - Rows absent from `SamplerKind` are intentionally non-executable and must fail at parse/spec resolution instead of reaching the native driver.
  - Current inventory-only examples include `ddim cfg++`, `dpm adaptive`, `dpm++ 2m sde heun`, `dpm++ 2m sde gpu`, `dpm++ 2m sde heun gpu`, `dpm++ sde`, `dpm++ 3m sde`, and `lcm`.
- Driver-owned stochastic determinism:
  - Native stochastic lanes use seeded `ImageRNG` through the driver.
  - No ambient randomness in native stochastic update paths.
- Runtime boundaries:
  - No imports from archived upstream snapshots into active runtime code.
  - If a required runtime invariant is missing (predictor data, malformed schedule, invalid options), fail loud.

## Risks / Invariants
- `steps >= 1`; sigma ladders must include a terminal zero.
- Predictors must expose finite sigma bounds (`sigma_min`, `sigma_max`).
- Predictor-backed ladder schedules require valid predictor ladder data (`predictor.sigmas` and related helpers when used).
- Sampling cancellation is honored by raising `RuntimeError("cancelled")` from the driver.

## Logging
- `CODEX_LOG_SAMPLER=1` enables sampler setup/step diagnostics.
- `CODEX_LOG_SIGMAS=1` logs compact sigma-ladder summaries.
- Sampler diagnostics do not alter routing; they are observability only.
