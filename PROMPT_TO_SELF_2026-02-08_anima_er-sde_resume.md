# Prompt to self — clean-session resume (ER-SDE / Anima)

```text
You are working in the repo `stable-diffusion-webui-codex`.
Continue from:
- CWD: /home/lucas/work/stable-diffusion-webui-codex
- Branch: master
- Last commit: c84f80bbf84ec0872e8f30bfc0a10f50cc5baab4
- Date (UTC-3): 2026-02-08T01:01:41-03:00

Objective (1 sentence)
- Implement complete ER-SDE support (`solver_type`, `max_stage`, `eta`, `s_noise`) and release it only for Anima.

State
- Done:
  - Comfy ER-SDE semantics mapped from `.refs/ComfyUI` (`sample_er_sde` + helper functions + custom-node options).
  - Plan drafted and frozen in `.sangoi/plans/2026-02-08-anima-er-sde-enable.md` (items 1-2 complete).
  - Partial backend edits already made:
    - `apps/backend/interfaces/api/routers/generation.py` has `_validate_er_sde_release_scope(...)`.
    - `apps/backend/types/payloads.py` now allows `extras.er_sde`.
  - `manage_context` cleanup was executed before handoff.
- In progress:
  - Runtime ER-SDE branch is not implemented yet in `apps/backend/runtime/sampling/driver.py`.
  - Option plumbing from API → processing plan/context → sampler is still missing.
- Blocked / risks:
  - Working tree is dirty with unrelated tracked changes (frontend + backend files). Keep patch scope tight.
  - Current Anima capability surface still excludes `er sde`; API tests currently expect Anima rejection.
  - User constraint: do not modify tests for now.

Decisions / constraints (locked)
- Support level: complete (`solver_type`, `max_stage`, `eta`, `s_noise`).
- Release scope: Anima only (non-Anima must fail loud with 400).
- API contract:
  - txt2img: `extras.er_sde`
  - img2img: `img2img_extras.er_sde`
  - same object applies to base + hires pass.
- `solver_type` accepts UI labels (`ER-SDE`, `Reverse-time SDE`, `ODE`) and canonical tokens.
- Keep fail-loud behavior and strict unknown-key rejection for ER-SDE option objects.
- Do not modify tests in this pass (user request).

Follow-up (ordered)
1. Implement native ER-SDE branch in `apps/backend/runtime/sampling/driver.py` using Comfy formulas/helpers.
2. Add ER-SDE option transport dataclass fields (`SamplingPlan`) and wiring (`sampling_plan.py` / `sampling_execute.py`).
3. Parse/validate `extras.er_sde` and `img2img_extras.er_sde` in `generation.py` with strict key/type checks.
4. Propagate parsed options through adapters/processing overrides to runtime sampler.
5. Enforce release guard: non-Anima + `er sde` => explicit 400 (base and hires fields).
6. Update Anima capability sampler allowlist to include `er sde` (keep default sampler unchanged).
7. Run compile + focused checks and update docs/task-log/changelog.

Next immediate step (do this first)
- Build the runtime ER-SDE helper layer in `driver.py` and wire `SamplerKind.ER_SDE` dispatch before touching broader API plumbing.
Commands:
CODEX_ROOT="$(git rev-parse --show-toplevel)"
export CODEX_ROOT PYTHONPATH="$CODEX_ROOT"
rg -n "SamplerKind\\.ER_SDE|NotImplementedError|CodexSampler|sample\\(" apps/backend/runtime/sampling/driver.py

Files
- Changed files (last commit(s)):
  - apps/backend/patchers/AGENTS.md
  - apps/backend/patchers/vae.py
  - apps/backend/runtime/families/anima/AGENTS.md
  - apps/backend/runtime/families/anima/wan_vae.py
  - apps/backend/runtime/families/wan22/AGENTS.md
  - apps/backend/runtime/families/wan22/wan_latent_norms.py
  - COMMON_MISTAKES.md
  - apps/backend/runtime/families/anima/loader.py
  - apps/backend/runtime/families/anima/text_encoder.py
  - apps/backend/runtime/state_dict/AGENTS.md
  - apps/backend/runtime/state_dict/__init__.py
  - apps/backend/runtime/state_dict/keymap_anima.py
- Focus files to open first:
  - apps/backend/runtime/sampling/driver.py — add native ER-SDE algorithm branch and helper math.
  - apps/backend/runtime/pipeline_stages/sampling_plan.py — carry ER-SDE options into `SamplingPlan`.
  - apps/backend/runtime/processing/datatypes.py — extend `SamplingPlan` with typed ER-SDE options.
  - apps/backend/runtime/pipeline_stages/sampling_execute.py — pass options into sampling context/driver path.
  - apps/backend/interfaces/api/routers/generation.py — strict parsing/validation for `extras.er_sde` and `img2img_extras.er_sde`.
  - apps/backend/types/payloads.py — keep extras whitelist aligned with ER-SDE contract.
  - apps/backend/engines/util/adapters.py — ensure overrides transport keeps ER-SDE object intact.
  - apps/backend/runtime/model_registry/capabilities.py — Anima-only release surface includes `er sde`.

Validation (what “green” looks like)
- `"$CODEX_ROOT/.venv/bin/python" -m py_compile apps/backend/runtime/sampling/driver.py apps/backend/runtime/pipeline_stages/sampling_plan.py apps/backend/interfaces/api/routers/generation.py apps/backend/types/payloads.py apps/backend/runtime/model_registry/capabilities.py`
  # expected: no syntax/import errors
- `python3 .sangoi/.tools/review_apps_header_updates.py --show-body-diff`
  # expected: touched `apps/**` files as `OK_HEADER_CHANGED` / `OK_NEW_FILE`
- `bash .sangoi/.tools/link-check.sh .sangoi`
  # expected: no broken markdown links

References (read before coding)
- .sangoi/plans/2026-02-08-anima-er-sde-enable.md
- .sangoi/task-logs/2026-02-07-anima-sampler-and-vae-decode-dtype-fix.md
- .sangoi/task-logs/2026-02-08-anima-vae-decode-wan21-parity.md
- .sangoi/CHANGELOG.md
- .refs/ComfyUI/comfy/k_diffusion/sampling.py
- .refs/ComfyUI/comfy/samplers.py
- .refs/ComfyUI/comfy_extras/nodes_custom_sampler.py
```
