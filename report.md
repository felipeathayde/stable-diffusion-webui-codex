# Subagent Responses Survey (`>= High/P1`)

Date: 2026-03-04  
Scope: `.sangoi/subagent-responses/**`  
Constraint applied: no new patrol subagents were spawned in this run.

## Snapshot

- Active pool (outside `archive/`):
  - `0` pending response files (root queue drained in latest archival batch)
- New-wave active files (`2026-03-03` and `2026-03-04`):
  - `0` pending files after archival consolidation
- Archive pool:
  - `809` archived response files
  - Buckets: `resolved` (`720`), `duplicates-2026-03-04` (`53`), `duplicates-2026-02-28` (`21`), `duplicates-2026-02-28-p1` (`15`)

## Execution Status (post-fan-in)

- Integrated execution commit: `459cb55ec` (`feat: integrate P0 runtime guards, policy matrix, WAN progress, and params migration`).
- Completed Code Machine responses from this execution wave were archived.
- Archive actions performed:
  - Moved to `archive/resolved`:
    - `senior-code-machine-response-20260304-155244Z.md`
    - `senior-code-machine-response-20260304-155352Z.md`
    - `senior-code-machine-response-20260304-155417Z.md`
    - `senior-code-machine-response-20260304-155736Z.md`
  - Moved to `archive/duplicates-2026-03-04` (superseded blocked/no-brief responses):
    - `senior-code-machine-response-20260304-154206Z.md`
    - `senior-code-machine-response-20260304-154244Z.md`
    - `senior-code-machine-response-20260304-154312Z.md`
    - `senior-code-machine-response-20260304-154334Z.md`
  - Moved to `archive/resolved` (next 12-lane execution wave):
    - `senior-code-machine-response-20260304-162256Z.md`
    - `senior-code-machine-response-20260304-162350Z.md`
    - `senior-code-machine-response-20260304-162354Z.md`
    - `senior-code-machine-response-20260304-162415Z.md`
    - `senior-code-machine-response-20260304-162416Z.md`
    - `senior-code-machine-response-20260304-162501Z.md`
    - `senior-code-machine-response-20260304-162553Z.md`
    - `senior-code-machine-response-20260304-162739Z.md`
    - `senior-code-machine-response-20260304-162839Z.md`
    - `senior-code-machine-response-20260304-162854Z.md`
    - `senior-code-machine-response-20260304-163055Z.md`
  - Moved to `archive/duplicates-2026-03-04` (next wave no-brief/superseded responses):
    - `senior-code-machine-response-20260304-161815Z.md`
    - `senior-code-machine-response-20260304-161817Z.md`
    - `senior-code-machine-response-20260304-161819Z.md`
    - `senior-code-machine-response-20260304-161824Z.md`
    - `senior-code-machine-response-20260304-161825Z.md`
    - `senior-code-machine-response-20260304-161826Z.md`
    - `senior-code-machine-response-20260304-161827Z.md`
    - `senior-code-machine-response-20260304-161828Z.md`
    - `senior-code-machine-response-20260304-161833Z.md`
  - Moved to `archive/resolved` (in-progress-cluster closure wave):
    - `senior-code-machine-response-20260304-164147Z.md`
    - `senior-code-machine-response-20260304-164429Z.md`
    - `senior-code-machine-response-20260304-164616Z.md`
    - `senior-code-machine-response-20260304-164645Z.md`
    - `senior-code-machine-response-20260304-164903Z.md`
  - Moved to `archive/duplicates-2026-03-04` (status-check/superseded responses):
    - `senior-code-machine-response-20260304-164430Z.md`
    - `senior-code-machine-response-20260304-164723Z.md`
    - `senior-code-machine-response-20260304-164726Z.md`
  - Moved to `archive/resolved` (remaining in-progress closure wave):
    - `senior-code-machine-response-20260304-165307Z.md`
    - `senior-code-machine-response-20260304-165802Z.md`
  - Moved to `archive/resolved` (bulk backlog consolidation):
    - `181` files moved (manifest: `.sangoi/subagent-responses/archive/archive-batch-20260304-190233Z.md`)
  - Moved to `archive/duplicates-2026-03-04` (bulk no-brief/readiness backlog):
    - `37` files moved (manifest: `.sangoi/subagent-responses/archive/archive-batch-20260304-190233Z.md`)

## Consolidated Active Findings (20 clusters, expanded)

### 1) Device policy inversion/misapplication across generation routes (includes Blocker)

- What is happening:
  - Image routes are being validated with WAN-like device constraints in some paths, while other routes still use broader canonical device resolution.
- Why this happens:
  - Validation ownership drifted to route-level branches with divergent allowlists/messages instead of one canonical policy source.
- Implication and impact:
  - Same device can be accepted/rejected depending on route, causing contract inconsistency, wrong error semantics, and user confusion.
- Resolution options:
  - A) Split allowlists per mode family and keep WAN-only restrictions scoped to WAN routes.
  - B) Centralize route/device policy in one matrix (`mode -> allowed devices`) used by all routers.
  - C) Validate by runtime capabilities resolved from engine+model, not by hardcoded route strings.
  - D) Add matrix contract checks (route x device) in validation gate to fail on drift.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-code-reviewer-response-20260303-144809Z.md:23`
  - `.sangoi/subagent-responses/archive/resolved/senior-code-reviewer-response-20260304-013709Z.md:25`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260303-210538Z.md:28`

### 2) Invalid RNG source is silently coerced (fail-loud violation)

- What is happening:
  - Unknown `randn_source`/noise-source values are accepted and silently remapped to fallback execution.
- Why this happens:
  - The plan builder uses permissive fallback logic rather than strict enum enforcement.
- Implication and impact:
  - Invalid payloads appear "successful", masking caller bugs and undermining reproducibility/debugging.
- Resolution options:
  - A) Enforce strict enum at API boundary and return HTTP 400 on invalid value.
  - B) Keep typed end-to-end field and include effective resolved source in diagnostics.
  - C) Remove fallback branch in planner; unknown value becomes fatal.
  - D) Constrain frontend emitters to server-advertised enum choices only.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-code-reviewer-response-20260304-013709Z.md:53`

### 3) Smart-cache contract errors are swallowed with hidden fallback behavior

- What is happening:
  - Text-encoder/cache-key path can swallow exceptions and continue with degraded behavior.
- Why this happens:
  - Broad exception handling is used where strict contract failures should be explicit.
- Implication and impact:
  - Cache may silently disable or misbehave; performance/correctness regressions become hard to attribute.
- Resolution options:
  - A) Replace broad exception swallowing with targeted exceptions + explicit fatal handling.
  - B) Return structured `Result` from cache-key path and fail loud on contract mismatch.
  - C) Classify errors (contract vs transient I/O) and only retry transient class.
  - D) Add telemetry counter + warning/error logs for every cache-key failure path.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-code-reviewer-response-20260303-205955Z.md:25`
  - `.sangoi/subagent-responses/archive/resolved/senior-code-reviewer-response-20260303-205955Z.md:43`

### 4) Core loading lifecycle remains eager in multiple families

- What is happening:
  - Core components are still realized eagerly across checkpoint and diffusers lanes.
- Why this happens:
  - No fully enforced lazy-load contract exists across all loaders; legacy eager seams remain.
- Implication and impact:
  - Higher VRAM peaks, slower startup, and fragile sequencing during refactor/migration.
- Resolution options:
  - A) Run phased migration per family with strict staging and rollback points.
  - B) Standardize lifecycle interface (`prepare/load/realize/unload`) for all loaders.
  - C) Move realization behind explicit runtime stage boundary with mandatory ownership.
  - D) Add conformity checks to block new eager paths from entering active code.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260304-011944Z.md:24`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260304-012219Z.md:31`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260304-012219Z.md:158`

### 5) Keymap/contract drift risk from hidden compatibility and ownership coupling

- What is happening:
  - Compatibility glue and keymap ownership are distributed, allowing divergent key semantics.
- Why this happens:
  - Transitional adapters/aliases were not fully removed after contract evolution.
- Implication and impact:
  - Strict validation can be bypassed indirectly; behavior differs by call path.
- Resolution options:
  - A) Remove hidden aliases in active seams and accept canonical keys only.
  - B) Consolidate canonical key equivalence in one registry with explicit ownership.
  - C) Generate allowed keys from schema at build time and reject undeclared keys at runtime.
  - D) Disable env-based bypasses in production profile.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260304-013759Z.md:32`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260304-013759Z.md:41`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260304-013759Z.md:70`

### 6) Anima lazy Qwen/VAE materialization occurs outside manager-owned load boundary

- What is happening:
  - Some heavyweight materialization occurs during run path, not manager-controlled load stage.
- Why this happens:
  - Ownership of full model realization is split between manager and model internals.
- Implication and impact:
  - VRAM accounting and latency become non-deterministic; offload policies lose predictability.
- Resolution options:
  - A) Force materialization inside manager-controlled load/transfer stage.
  - B) Add manager preflight reservation/checks for all deferred components.
  - C) Require explicit model callbacks for "about to realize heavy tensors".
  - D) Block execution when manager has not acknowledged all required realized components.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260303-225029Z.md:27`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260303-225029Z.md:49`

### 7) Unload path reports success while residency can remain

- What is happening:
  - Unload operation can report success even when model residency is not fully released.
- Why this happens:
  - Logical unload state and real tensor/object residency are not fully coupled.
- Implication and impact:
  - Memory leaks/residency drift accumulate and reduce availability for subsequent runs.
- Resolution options:
  - A) Add post-unload residency assertions and fail loud on mismatch.
  - B) Maintain a single authoritative residency state machine.
  - C) Add explicit cleanup audit (refs/tensors/modules) after unload stage.
  - D) Expose diagnostic endpoint/trace for runtime residency verification.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260303-230013Z.md:11`

### 8) SDPA route rule ambiguity around `head_dim` threshold

- What is happening:
  - The current behavior/policy is interpreted inconsistently: flash attention expectation and fallback route behavior are mixed as if they were the same rule.
- Why this happens:
  - The rule was framed as coercion logic instead of an explicit threshold policy (`<= 256` vs `> 256`) with deterministic route ownership.
- Implication and impact:
  - Runtime choice can look arbitrary, making performance and debugging unpredictable.
- Resolution options:
  - A) Lock the rule explicitly: `head_dim <= 256` => always flash when available; `head_dim > 256` => default non-flash route.
  - B) Resolve backend policy through capability matrix (`policy x head_dim x backend`).
  - C) Attempt requested policy first and fallback only on proven incompatibility.
  - D) Add strict mode that errors instead of silently coercing policy.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260304-023644Z.md:34`

### 9) Bootstrap splash can stall indefinitely on hanging required requests

- What is happening:
  - Required startup requests can hang and keep splash locked without terminal state.
- Why this happens:
  - Bootstrap gate has no timeout/abort policy for required paths.
- Implication and impact:
  - App appears dead/stuck even when backend is partially healthy.
- Resolution options:
  - A) Add timeout + abort for required bootstrap requests.
  - B) Split bootstrap into minimal ready stage plus deferred background stage.
  - C) Add watchdog timer per stage with explicit fatal reason.
  - D) Use short-lived startup cache to avoid hard dependency on slow calls.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260303-175353Z.md:24`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260303-175353Z.md:34`

### 10) VAE selection persistence is vulnerable to overwrite/canonicalization drift

- What is happening:
  - Persisted VAE can be overwritten by fallback canonicalization in non-target context.
- Why this happens:
  - One global storage key is used while family resolution can be transient/ambiguous.
- Implication and impact:
  - User selection is silently lost across reloads/context switches.
- Resolution options:
  - A) Persist VAE selection per family, not globally.
  - B) Move persistence to server-side profile/tab state.
  - C) Do not persist automatic fallback; persist only explicit user actions.
  - D) Add precedence model where user-selected value outranks canonical fallback writes.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260303-231229Z.md:26`

### 11) LoRA refresh path is synchronous while async task/SSE handling has sharp edges

- What is happening:
  - Modal still uses sync refresh path despite existing async task and SSE substrate.
- Why this happens:
  - Integration was not migrated end-to-end (start task + subscribe + terminal apply).
- Implication and impact:
  - UI blocks on long scans; stale states/errors occur when refresh lifecycle is partial.
- Resolution options:
  - A) Migrate modal refresh to async task + SSE terminal apply.
  - B) Introduce centralized frontend task orchestrator with retry/cancel/timeout.
  - C) Add result versioning so only latest refresh updates modal/store.
  - D) Buffer payload and apply only on terminal event with strict error handling.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260303-173332Z.md:14`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260303-173332Z.md:28`

### 12) QuickSettings add-path byte-progress contract is incomplete

- What is happening:
  - `size_bytes` coverage and progress totals are incomplete/inconsistent through scan/add flow.
- Why this happens:
  - Contract did not standardize byte metadata availability and recomputation semantics.
- Implication and impact:
  - "Honest byte-progress" can degrade to spinner mode or partial truth despite visible size labels.
- Resolution options:
  - A) Add `size_bytes` to scan/add contract and propagate through rows.
  - B) Recompute planned totals when new size metadata appears during add-all.
  - C) Define explicit row FSM (`queued/adding/done/error`) with byte semantics.
  - D) Add dedicated progress payload/update path instead of UI-derived inference.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260303-181117Z.md:16`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260303-191035Z.md:14`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260303-200737Z.md:10`

### 13) Video overlay ownership/trigger wiring and media interaction suppression are high-risk seams

- What is happening:
  - Overlay triggering and interaction suppression are split across seams with inconsistent behavior.
- Why this happens:
  - Ownership of overlay state and media events is distributed across UI layers.
- Implication and impact:
  - Accidental opens/closes, blocked media controls, and unreliable user interaction.
- Resolution options:
  - A) Restrict overlay trigger to explicit video-result action path.
  - B) Consolidate overlay behavior in a dedicated video overlay component.
  - C) Separate gesture surface from controls with explicit pointer-event policy.
  - D) Add interaction contract tests for click/dblclick/keyboard paths.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260304-031721Z.md:22`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260304-031721Z.md:42`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260304-031721Z.md:69`

### 14) App mode/launcher integration is not a narrow toggle

- What is happening:
  - Backend can serve SPA, but launcher/dev-service assumptions are not harmonized.
- Why this happens:
  - Feature touches multiple layers (backend mode, launcher UX, packaging/paths, startup expectations).
- Implication and impact:
  - Partial enablement causes startup/runtime contradictions and hard-to-debug failures.
- Resolution options:
  - A) Add preflight checks before enabling app mode.
  - B) Split launcher paths into explicit mode profiles (dev-service vs embedded).
  - C) Enforce release packaging contract for SPA assets before mode switch.
  - D) Provide explicit fallback path with clear user-facing diagnostics.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260304-035733Z.md:19`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260304-035733Z.md:30`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260304-035733Z.md:87`

### 15) Quicksettings model list has multiple non-folder gates and drift vectors

- What is happening:
  - Model visibility depends on endpoint split, cache state, filters, dedupe, and scanner policy differences.
- Why this happens:
  - Legacy checkpoint chain and inventory chain evolved separately with separate invalidation semantics.
- Implication and impact:
  - Files present on disk can still not appear (or appear stale), confusing operator expectations.
- Resolution options:
  - A) Standardize refresh/invalidation semantics across both list chains.
  - B) Unify list source into one normalized inventory pipeline.
  - C) Introduce freshness token/version so frontend detects stale cache deterministically.
  - D) Emit scanner/list diagnostics to explain why each candidate was filtered.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260302-203325Z.md:33`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260302-203325Z.md:48`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260302-203325Z.md:62`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260302-203325Z.md:74`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260302-203325Z.md:87`

### 16) Strict unknown-params backend guard clashes with persisted legacy keys

- What is happening:
  - Backend now rejects unknown top-level params while frontend can persist/resend legacy keys.
- Why this happens:
  - Local persisted schema migration is incomplete after backend strictness change.
- Implication and impact:
  - Repeated 400s for users with stale local state until manual cleanup/migration.
- Resolution options:
  - A) Sanitize outgoing payload against current schema before submit.
  - B) Version and migrate persisted `tab.params` during load.
  - C) Return structured unknown-key diagnostics to aid precise cleanup.
  - D) Run one-shot local storage migration on startup with fail-loud notice.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260301-172456Z.md:17`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260301-172456Z.md:30`

### 17) Unified block-progress contract is partially disconnected in WAN paths

- What is happening:
  - Some WAN flow paths do not emit unified block-level progress as expected.
- Why this happens:
  - Progress callback wiring is incomplete across all execution branches.
- Implication and impact:
  - UI progress semantics drift from actual execution stages.
- Resolution options:
  - A) Wire unified callback across every WAN execution branch.
  - B) Introduce adapter contract per family with mandatory progress hooks.
  - C) Add runtime assertion when stage executes without progress emitter.
  - D) Provide explicit "coarse progress only" state where fine-grain is unavailable.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260301-053432Z.md:17`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260301-172129Z.md:31`

### 18) WAN video toggles/flags and memory path still have high-risk mismatches

- What is happening:
  - Some video toggles are not reflected with full truth in effective payload path; SeedVR2 path can materialize heavy sequences.
- Why this happens:
  - UI/config toggles and execution plan mapping are not fully harmonized, and memory planning is too eager in parts.
- Implication and impact:
  - Output semantics differ from UI expectation; memory spikes increase OOM risk.
- Resolution options:
  - A) Fix toggle-to-payload mapping and expose effective execution settings.
  - B) Switch heavy video paths to chunked/streamed processing where possible.
  - C) Add immutable request-vs-effective snapshot in task metadata.
  - D) Enforce memory budget guardrails before sequence materialization.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260301-225104Z.md:31`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260301-225104Z.md:59`

### 19) Global Hires prep remains SD-family hard-wired

- What is happening:
  - Hires prep path assumes SD-family backend primitives as canonical.
- Why this happens:
  - Shared stage abstraction did not separate family-specific prep strategy.
- Implication and impact:
  - Non-SD families are blocked or forced through brittle compatibility paths.
- Resolution options:
  - A) Gate unsupported families explicitly with fail-loud errors.
  - B) Refactor Hires prep into family strategy hooks.
  - C) Introduce strategy registry keyed by family.
  - D) Align UI capability exposure with backend strategy availability.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260302-174101Z.md:72`

### 20) VRAM/offload observability and trace/debug behavior has known high-risk masking paths

- What is happening:
  - Offload/fallback/trace decisions are not fully transparent at runtime.
- Why this happens:
  - Logging/telemetry granularity is uneven and some fallbacks are silent.
- Implication and impact:
  - Performance regressions and memory path issues are hard to root-cause quickly.
- Resolution options:
  - A) Add structured logs for offload/fallback decisions by stage.
  - B) Define telemetry contract per pipeline stage and enforce emission.
  - C) Introduce end-to-end correlation id across route/use-case/engine path.
  - D) Add alert thresholds for fallback rate and VRAM anomaly spikes.
- Evidence:
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260301-035928Z.md:21`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260301-172751Z.md:27`
  - `.sangoi/subagent-responses/archive/resolved/senior-scout-cavalry-response-20260301-221006Z.md:27`
  - `.sangoi/subagent-responses/archive/resolved/senior-code-reviewer-response-20260301-221949Z.md:22`
  - `.sangoi/subagent-responses/archive/resolved/senior-code-reviewer-response-20260301-172541Z.md:27`

## Locked Decision Matrix (selected options + delivery priority)

| Item | Selected options | Priority | Locked implementation note |
| --- | --- | --- | --- |
| 1 | **B** | P0 | Implement route/device policy as typed matrix (dataclass/enum owned), single authority. |
| 2 | **A** | P0 | Reject invalid RNG source at boundary; no silent coercion. |
| 3 | **B** | P0 | Structured result path; contract mismatch must fail loud. |
| 4 | **B+C+D** | P0 | Enforce global load authority matrix + runtime wrapper + hard crash on violation. |
| 5 | **A+B** | P1 | Remove hidden aliases and centralize canonical key equivalence ownership. |
| 6 | **A+D** | P0 | Manager-owned materialization + global block for non-acknowledged load attempts. |
| 7 | **A+B** | P0 | Verified unload + authoritative residency state machine. |
| 8 | **A** (reframed rule) | P0 | `head_dim <= 256` => flash when available; `head_dim > 256` => default non-flash path. |
| 9 | **B** | P1 | Two-phase bootstrap (minimal-ready + deferred heavy init), not timeout-driven. |
| 10 | **A+B** | P1 | Family-scoped persistence plus server profile/tab persistence for stability. |
| 11 | **A+B** | P1 | Async task/SSE refresh with centralized task orchestrator. |
| 12 | **A+C+D** | P1 | `size_bytes` + explicit row FSM + dedicated progress update path. |
| 13 | **A+B+C** | P1 | Explicit trigger + dedicated overlay component + strict pointer-event zoning. |
| 14 | **A+B+C** | P2 | App-mode preflight + launcher mode split + packaging contract enforcement. |
| 15 | **A+B+C** | P0 | Unified model-list source + shared invalidation + freshness/version token. |
| 16 | **B+C** | P0 | Versioned migration of persisted params + structured unknown-key diagnostics. |
| 17 | **B+C** | P0 | Family adapter with mandatory hooks + runtime assertion on missing progress emitter. |
| 18 | **A+C** | P1 | Correct toggle mapping + immutable request-vs-effective execution snapshot. |
| 19 | **B+C** (if compatible) | P1 | Hires strategy hooks + registry; enforce compatibility gate and fail loud when incompatible. |
| 20 | **A+B+C** | P1 | Structured logs + stage telemetry contract + end-to-end correlation id. |

### Item 4 — Load Authority Priority Matrix (mandatory, fail-loud)

**Policy intent:** loading authority is centralized. Any actor outside allowed authority that performs a wrapped load action must **explode with error**.

| Situation | Allowed authority to load | Required permit/wrap | Forbidden actors | Violation behavior |
| --- | --- | --- | --- | --- |
| Process startup prewarm | `RuntimeLoadCoordinator` only | `LoadPermit(stage=startup, owner=runtime)` | Routers, UI-facing handlers, engine adapters | `RuntimeError(LOAD_AUTHORITY_VIOLATION)` |
| Request preflight (`ensure assets/load`) | Canonical mode use-case via `RuntimeLoadCoordinator` | `LoadPermit(stage=preflight, owner=use_case)` | Routers direct-load, model adapters direct-load | `RuntimeError(LOAD_AUTHORITY_VIOLATION)` |
| Pipeline execution-stage realization | Pipeline stage through `RuntimeLoadCoordinator` ack path | `LoadPermit(stage=execute, owner=pipeline_stage)` | Any ad-hoc lazy materialization in module `forward()` without permit | `RuntimeError(LOAD_AUTHORITY_VIOLATION)` |
| Async/background metadata tasks | No heavyweight model load allowed | Metadata-only path (no load permit issued) | Background scanners triggering tensor/model load | `RuntimeError(LOAD_AUTHORITY_VIOLATION)` |
| Reload/unload transitions | `RuntimeLoadCoordinator` state machine | `LoadPermit(stage=reload/unload, owner=runtime)` | Local cleanup helpers changing residency without coordinator | `RuntimeError(LOAD_AUTHORITY_VIOLATION)` |

**Global runtime wrap (locked):**
- Wrap high-risk load entrypoints in a global guard layer under runtime ownership.
- Guard requires an active `LoadPermit` token (typed enum context).
- No token or wrong owner/stage => immediate hard failure (`LOAD_AUTHORITY_VIOLATION`), no fallback path.

**Global non-acknowledged block (linked to Item 6 A+D):**
- Any load/materialization attempt not acknowledged by runtime coordinator is blocked globally.
- This includes deferred/lazy paths reached during execution without explicit permit.

## Cluster Execution Status (current)

Status source: integrated fan-in commit `459cb55ec` + completed 12-lane wave + in-progress-cluster closure wave (2026-03-04).

| Cluster | Status | Scope note |
| --- | --- | --- |
| 1 | **Done** | Typed route/device policy matrix is integrated in backend route parsing flow. |
| 2 | **Done** | Invalid `randn_source` is rejected at boundary and planner no longer silently coerces invalid values. |
| 3 | **Done** | Smart-cache/runtime-override handling is now strict fail-loud with classified errors and no silent swallow path. |
| 4 | **Done** | Runtime load authority permits/guards are implemented with fail-loud violations. |
| 5 | **Done** | WAN canonical key ownership is centralized and legacy aliases are rejected at generation seams. |
| 6 | **Done** | Non-acknowledged load/materialization is globally blocked by authority checks. |
| 7 | **Done** | Engine-loader cleanup now follows fail-loud unload parity with explicit post-unload residency verification (no suppress-based cleanup masking). |
| 8 | **Done** | SDPA threshold rule is explicit (`<=256` flash-preferred, `>256` non-flash route). |
| 9 | **Done** | Bootstrap now uses two-phase readiness (`critical` + deferred heavy init) instead of a single blocking gate. |
| 10 | **Done** | Family-scoped VAE map is now persisted server-side via validated backend option contract (`codex_vae_by_family`) with rollback on apply failure. |
| 11 | **Done** | LoRA refresh+hydrate orchestration is centralized in store authority and reused by QuickSettings and LoRA modal (duplicate local orchestration removed). |
| 12 | **Done** | Add-path flow now has explicit row FSM, strict `size_bytes` contract, and deterministic byte-progress behavior. |
| 13 | **Done** | Overlay flow now uses explicit trigger ownership + dedicated component zoning for interaction safety. |
| 14 | **Done** | App-mode preflight/profile split/packaging contract enforcement is integrated in launcher + backend boot path. |
| 15 | **Done** | Backend model catalog now has unified refresh/invalidation authority and shared `models_revision` freshness token. |
| 16 | **Done** | Persisted params versioned migration + structured unknown-key diagnostics are integrated. |
| 17 | **Done** | WAN unified block-progress adapter + required emitter assertions are integrated. |
| 18 | **Done** | WAN output toggles are normalized end-to-end: payload mapping uses real request intent and WAN UI now exposes explicit controls for `saveOutput/saveMetadata/trimToAudio`. |
| 19 | **Done** | Hires compatibility strategy seam and fail-loud family gate are integrated in shared runtime/use-case paths. |
| 20 | **Done** | Canonical task-scoped correlation context + structured run/stage telemetry now cover txt2img/img2img/txt2vid/img2vid/vid2vid stage boundaries. |

## Historical `>= High/P1` (Archive)

- Archive currently contains `809` response files across resolved/duplicate buckets.
- The dominant buckets are `resolved` and `duplicates`; they remain useful as incident history but should not be treated as active blockers without revalidation.

Examples of historical P1/Blocker references:
- `.sangoi/subagent-responses/archive/resolved/senior-code-reviewer-response-20260225-175321Z.md:2`
- `.sangoi/subagent-responses/archive/resolved/senior-code-reviewer-response-20260225-213512Z.md:2`
- `.sangoi/subagent-responses/archive/resolved/senior-code-reviewer-response-20260228-211821Z.md:33`

## Notes

- This report deduplicates by **problem cluster**, not by raw hit count.
- Several files in the 2026-03-03/04 wave include Medium/Hypothesis findings not included here due the requested cut (`>= High/P1`).
