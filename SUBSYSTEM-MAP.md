<!-- tags: webui, architecture, map, discovery, backend -->
# WebUI Subsystem Map
Date: 2026-04-06
Last Review: 2026-04-06
Status: Active

## Purpose / ownership boundary
- This file is a manual discovery atlas.
- It stays discovery-only: hotspot directory, bounded pipeline node chains, owner seams, and pointers to generated artifacts/policy.
- It is not the owner for contract matrices, migration ledgers, or mutable backend counts/timestamps.
- Current owner split:
  - `SUBSYSTEM-MAP-INDEX.md` = lookup-only front door
  - `SUBSYSTEM-MAP.md` = hotspot directory + pipeline node atlas + owner seam atlas
  - `AGENTS.md` = maintenance triggers and same-tranche sync workflow
  - `.sangoi/policies/file-header-block.md` = file-header content standard
  - generated reports under `.sangoi/reports/tooling/` = mutable backend snapshot/count/timestamp owners

## How to use the index + map
1. Start in `SUBSYSTEM-MAP-INDEX.md`.
2. Use the `Jump to` link for the hotspot, pipeline, or owner seam you need.
3. From the atlas section below, open the canonical owner path first.
4. Only after that, widen into secondary seams or generated reports.

## Hotspot directory
<a id="map-hotspot-keymaps"></a>
### Keymaps
- Open first: `apps/backend/runtime/state_dict/key_mapping.py`
- Secondary seams:
  - `apps/backend/runtime/state_dict/keymap_anima_transformer.py`
  - `apps/backend/runtime/state_dict/keymap_wan22_transformer.py`
  - `apps/backend/runtime/state_dict/AGENTS.md`
- Use this when you are mapping upstream checkpoint keyspaces into the canonical runtime lookup view without rewriting stored keys, including raw Anima core checkpoints stored under `net.*`.

<a id="map-hotspot-vae-codex3d"></a>
### Native Conv3D VAE
- Open first: `apps/backend/runtime/common/vae_codex3d.py`
- Secondary seams:
  - `apps/backend/runtime/families/wan22/vae_io.py`
  - `apps/backend/runtime/state_dict/keymap_wan22_vae.py`
- Use this when a WAN-style native 3D VAE path, shift/scale policy, or load/detect question points at the codex 3D lane.

<a id="map-hotspot-hires-fix"></a>
### Hires Fix
- Open first: `apps/backend/runtime/pipeline_stages/hires_fix.py`
- Secondary seams:
  - `apps/backend/runtime/families/sd/hires_fix.py`
  - `apps/backend/use_cases/img2img.py`
- Use this when the second pass, hires geometry, continuation semantics, or hires telemetry ownership is the real seam.

<a id="map-hotspot-image-automation"></a>
### Image Automation
- Open first: `apps/backend/use_cases/image_automation.py`
- Secondary seams:
  - `apps/backend/interfaces/api/tasks/generation_tasks.py`
  - `apps/backend/interfaces/api/routers/generation.py`
- Use this when a repeat/infinite run, prompt-list/wildcard expansion, folder source cycling, or automation summary issue is really backend-owned.

<a id="map-hotspot-ip-adapter"></a>
### IP-Adapter
- Open first: `apps/backend/runtime/pipeline_stages/ip_adapter.py`
- Secondary seams:
  - `apps/backend/runtime/adapters/ip_adapter/`
  - `apps/backend/interfaces/api/routers/generation.py`
- Use this when adapter model/image-encoder selection, reference-image preprocessing, or per-sampling patch application is the seam.

<a id="map-hotspot-supir-runtime"></a>
### SUPIR Runtime
- Open first: `apps/backend/runtime/families/supir/runtime.py`
- Secondary seams:
  - `apps/backend/use_cases/img2img.py`
  - `apps/backend/interfaces/api/routers/generation.py`
  - `apps/backend/runtime/families/supir/loader.py`
- Use this when native SDXL img2img/inpaint SUPIR mode, restore-anchor sampling, or SUPIR variant/base-checkpoint validation is the seam.

<a id="map-hotspot-attention-sram-v1"></a>
### attention_sram_v1
- Open first: `apps/backend/runtime/attention/sram/__init__.py`
- Secondary seams:
  - `apps/backend/runtime/kernels/attention_sram_v1/`
  - `apps/backend/runtime/families/wan22/model.py`
- Use this when you are debugging the runtime bridge, extension build/warmup, split-KV heuristics, or the shared-memory kernel lane.

<a id="map-hotspot-generation-router"></a>
### Generation Router
- Open first: `apps/backend/interfaces/api/routers/generation.py`
- Secondary seams:
  - `apps/backend/interfaces/api/tasks/generation_tasks.py`
  - `apps/backend/interfaces/api/run_api.py`
  - `apps/backend/runtime/logging.py`
- Use this when a public generation route, payload parser, task spawn, or route-level fail-loud guard is the true owner.

<a id="map-hotspot-task-streaming"></a>
### Task Streaming
- Open first: `apps/backend/interfaces/api/routers/tasks.py`
- Secondary seams:
  - `apps/backend/interfaces/api/task_registry.py`
  - `apps/backend/interfaces/api/tasks/generation_tasks.py`
- Use this when the question is task snapshots, SSE replay/gap recovery, terminal result/end emission, or cancellation semantics.

<a id="map-hotspot-orchestrator"></a>
### Orchestrator
- Open first: `apps/backend/core/orchestrator.py`
- Secondary seams:
  - `apps/backend/core/engine_interface.py`
  - `apps/backend/engines/common/base.py`
- Use this when engine resolution, load/unload/cache ownership, or wrapper dispatch is the real middle node.

## Pipeline node atlas
<a id="map-pipeline-txt2img"></a>
### txt2img
| Node | Owner | What happens here | Next |
| --- | --- | --- | --- |
| Public route | `apps/backend/interfaces/api/routers/generation.py` (`/api/txt2img`) | Validates payload, route capability, and explicit device; creates `TaskEntry`; starts the task thread. | `run_txt2img_task` |
| Shared image task worker | `apps/backend/interfaces/api/tasks/generation_tasks.py` | Calls `prepare_txt2img(...)`, owns inference-gate/task lifecycle, and reuses shared image result packaging. | `InferenceOrchestrator.run(...)` |
| Orchestrator | `apps/backend/core/orchestrator.py` | Resolves engine registry/load/cache state and dispatches to the engine wrapper. | engine `txt2img(...)` wrapper |
| Engine wrapper | `apps/backend/engines/common/base.py` | Delegates the mode to the canonical use-case instead of owning a second pipeline. | `run_txt2img(...)` |
| Canonical use-case | `apps/backend/use_cases/txt2img.py` | Owns progress/result emission, decode, and cleanup inside the worker-thread envelope. | `generate_txt2img(...)` |
| Stage runner | `apps/backend/use_cases/txt2img_pipeline/runner.py` | Executes the staged txt2img pipeline and returns the `GenerationResult`. | API result packaging |
| Terminal surfaces | `apps/backend/interfaces/api/tasks/generation_tasks.py` and `apps/backend/interfaces/api/routers/tasks.py` | Encodes/saves images, stores the result payload, and exposes terminal result/end through `GET /api/tasks/{id}` and `/api/tasks/{id}/events`. | end |

<a id="map-pipeline-img2img"></a>
### img2img
| Node | Owner | What happens here | Next |
| --- | --- | --- | --- |
| Public route | `apps/backend/interfaces/api/routers/generation.py` (`/api/img2img`) | Validates payload + route capability, rejects masked requests when the semantic engine capability surface says img2img masking is unsupported, validates exact-engine `img2img_inpaint_mode`, preflights native `img2img_extras.supir` and exact SDXL Fooocus/BrushNet assets, then creates the task and picks the explicit device. | shared image task worker |
| Shared image task worker | `apps/backend/interfaces/api/tasks/generation_tasks.py` | Calls `prepare_img2img(...)`, owns inference-gate/task lifecycle, and packages the terminal image result. | orchestrator |
| Orchestrator | `apps/backend/core/orchestrator.py` | Resolves engine/load/cache state and dispatches to the mode wrapper. | engine `img2img(...)` wrapper |
| Engine wrapper | `apps/backend/engines/common/base.py` | Delegates to the canonical img2img use-case. | `run_img2img(...)` |
| Canonical use-case | `apps/backend/use_cases/img2img.py` | Owns classic-family dispatch, init-image planning, prompt/sampling plans, optional native SUPIR mode, optional masked img2img, exact-engine SDXL Fooocus/BrushNet branching, and optional hires continuation. | shared stage helpers + sampler |
| Shared stage helpers | `apps/backend/runtime/pipeline_stages/masked_img2img.py`, `apps/backend/runtime/families/sd/fooocus_inpaint.py`, `apps/backend/runtime/families/sd/brushnet.py`, and `apps/backend/runtime/pipeline_stages/hires_fix.py` | Prepare generic masked bundles/image conditioning/hires latents, while `fooocus_inpaint.py` and `brushnet.py` own the request-scoped SDXL exact-engine patch sessions before masked sampling. | API result packaging |
| Terminal surfaces | `apps/backend/interfaces/api/tasks/generation_tasks.py` and `apps/backend/interfaces/api/routers/tasks.py` | Store the encoded result payload and expose terminal snapshot/SSE state. | end |

Branch notes:
- Classic base img2img resolves SD-vs-flow dispatch locally in `apps/backend/use_cases/img2img.py` before masked/unmasked prep.
- SDXL SUPIR mode stays inside the canonical img2img owner: route preflight lives in `apps/backend/interfaces/api/routers/generation.py`, while the request-scoped restore runtime lives in `apps/backend/runtime/families/supir/runtime.py`.
- SDXL exact-engine inpaint stays inside the canonical img2img owner: exact-engine mode/asset preflight lives in `apps/backend/interfaces/api/routers/generation.py`, while the request-scoped patch sessions live in `apps/backend/runtime/families/sd/fooocus_inpaint.py` and `apps/backend/runtime/families/sd/brushnet.py`.
- Kontext-specific img2img work stays local to `apps/backend/use_cases/img2img.py`.
- FLUX.2 keeps its own engine-side img2img seam at `apps/backend/engines/flux2/img2img.py`; the public route still enters through the same router/task/orchestrator chain.

<a id="map-pipeline-image-automation"></a>
### image_automation
| Node | Owner | What happens here | Next |
| --- | --- | --- | --- |
| Public route | `apps/backend/interfaces/api/routers/generation.py` (`/api/image-automation`) | Parses the backend-owned automation contract, selects the explicit device from the template payload, and creates the task. | `run_image_automation_task(...)` |
| Automation task worker | `apps/backend/interfaces/api/tasks/generation_tasks.py` | Owns task lifecycle, inference gate, per-iteration execution wrapper, automation summary, and terminal task result storage. | `run_image_automation(...)` |
| Automation loop owner | `apps/backend/use_cases/image_automation.py` | Expands prompts/wildcards, selects folder/init/reference inputs, manages loop cancellation, and emits iteration/progress events. | per-iteration delegate |
| Iteration delegate | `apps/backend/interfaces/api/tasks/generation_tasks.py` (`_execute_prepared_image_request`) | Turns each prepared iteration back into the canonical txt2img/img2img request path. | canonical image pipeline |
| Underlying mode | `apps/backend/use_cases/txt2img.py` or `apps/backend/use_cases/img2img.py` | Executes the actual generation for one iteration. | automation summary/result |
| Terminal surfaces | `apps/backend/interfaces/api/tasks/generation_tasks.py` and `apps/backend/interfaces/api/routers/tasks.py` | Publish `automation_iteration` events, store the final automation summary/result, and expose replay through task snapshot/SSE. | end |

<a id="map-pipeline-txt2vid"></a>
### txt2vid
| Node | Owner | What happens here | Next |
| --- | --- | --- | --- |
| Public route | `apps/backend/interfaces/api/routers/generation.py` (`/api/txt2vid`) | Validates payload/capability, rejects legacy aliases, creates the task, and selects the explicit device. | `run_video_task(...)` |
| Shared video task worker | `apps/backend/interfaces/api/routers/generation.py` | Parses the request, owns task lifecycle for video modes, and stores the terminal result payload. | orchestrator |
| Orchestrator | `apps/backend/core/orchestrator.py` | Resolves engine/load/cache state and dispatches to the engine wrapper. | engine `txt2vid(...)` wrapper |
| Engine wrapper | `apps/backend/engines/ltx2/ltx2.py` or `apps/backend/engines/wan22/wan22_14b.py` | Delegates to the canonical use-case for the active engine family. | `run_txt2vid(...)` |
| Canonical use-case | `apps/backend/use_cases/txt2vid.py` | Owns execution-profile branching, shared video plan/export helpers, optional upscaling/interpolation, and terminal `ResultEvent` emission. | export/result packaging |
| Shared video helpers | `apps/backend/runtime/pipeline_stages/video.py` | Owns `build_video_plan(...)`, `build_ltx2_video_plan(...)`, WAN Diffusers stage-LoRA preflight/apply, export helpers, and post-generation video stages. | task result storage |
| Terminal surfaces | `apps/backend/interfaces/api/routers/generation.py` and `apps/backend/interfaces/api/routers/tasks.py` | Store the final result in the task entry and expose terminal snapshot/SSE state. | end |

Branch notes:
- LTX2 keeps its execution-profile branch inside `apps/backend/use_cases/txt2vid.py`.
- WAN22 keeps the GGUF/diffusers branch decisions inside the same use-case after the wrapper/orchestrator hand-off.

<a id="map-pipeline-img2vid"></a>
### img2vid
| Node | Owner | What happens here | Next |
| --- | --- | --- | --- |
| Public route | `apps/backend/interfaces/api/routers/generation.py` (`/api/img2vid`) | Validates payload/capability, applies route-specific preflight, creates the task, and selects the explicit device. | `run_video_task(...)` |
| Shared video task worker | `apps/backend/interfaces/api/routers/generation.py` | Parses the request, owns task lifecycle, and stores the terminal result payload. | orchestrator |
| Orchestrator | `apps/backend/core/orchestrator.py` | Resolves engine/load/cache state and dispatches to the engine wrapper. | engine `img2vid(...)` wrapper |
| Engine wrapper | `apps/backend/engines/ltx2/ltx2.py` or `apps/backend/engines/wan22/wan22_14b.py` | Delegates to the canonical use-case for the active engine family. | `run_img2vid(...)` |
| Canonical use-case | `apps/backend/use_cases/img2vid.py` | Owns image-video execution profiles, WAN temporal-mode branching, shared video plan/export helpers, and terminal `ResultEvent` emission. | export/result packaging |
| Shared video helpers | `apps/backend/runtime/pipeline_stages/video.py` | Owns plan/export/upscale/interpolation helpers shared with txt2vid plus WAN Diffusers stage-LoRA preflight/apply. | task result storage |
| Terminal surfaces | `apps/backend/interfaces/api/routers/generation.py` and `apps/backend/interfaces/api/routers/tasks.py` | Store the final result in the task entry and expose terminal snapshot/SSE state. | end |

<a id="map-pipeline-vid2vid"></a>
### vid2vid
| Node | Owner | What happens here | Next |
| --- | --- | --- | --- |
| Public route | `apps/backend/interfaces/api/routers/generation.py` (`/api/vid2vid`) | Current public truth: the route is parked and fails fast with `400` before staging/task creation while no families are implemented. | end |
| Dormant wrapper | `apps/backend/engines/wan22/wan22_14b.py` | In-tree wrapper hook still exists for the future re-enable path. | dormant use-case |
| Dormant use-case | `apps/backend/use_cases/vid2vid.py` | Holds the bounded vid2vid execution owner once the public route is re-enabled. | dormant terminal result |

<a id="map-pipeline-api-bootstrap"></a>
### API bootstrap
| Node | Owner | What happens here | Next |
| --- | --- | --- | --- |
| App bootstrap | `apps/backend/interfaces/api/run_api.py` | Builds the FastAPI app, validates startup/runtime settings, and mounts the routers. Repo-owned bootstrap/server logs consume the canonical wrapper family from `apps/backend/runtime/logging.py`, while the remaining raw logger carve-out is the explicit `uvicorn.access` integration seam. | router module build |
| Router mount | `apps/backend/interfaces/api/run_api.py` | Includes `system`, `settings`, `ui`, `models`, `paths`, `options`, `tasks`, `tests`, `tools`, `upscale`, `supir`, and `generation`; the `supir` router is diagnostics-only. | public routes |
| Public entry | router modules under `apps/backend/interfaces/api/routers/` | Expose the task-backed generation/system/tool surfaces. | task or direct route handling |

## Owner seam atlas
<a id="map-owner-generation-router"></a>
### Generation Router seam
- Canonical owner: `apps/backend/interfaces/api/routers/generation.py`
- Owns:
  - public generation routes
  - payload parsing and route-level capability guards
  - exact-engine SUPIR-mode preflight for canonical img2img/inpaint
  - task creation and worker thread hand-off
- Do not move mode execution into this file; it should stay validate + dispatch + stream.

<a id="map-owner-image-task-worker"></a>
### Image Task Worker
- Canonical owner: `apps/backend/interfaces/api/tasks/generation_tasks.py`
- Owns:
  - shared image task lifecycle
  - inference-gate integration
  - encoded image result packaging and save/provenance hooks
  - automation task wrapper around canonical image modes
- Open this file when the question is task result packaging rather than public payload parsing.

<a id="map-owner-task-registry-sse"></a>
### Task Registry + SSE
- Canonical owners:
  - `apps/backend/interfaces/api/task_registry.py`
  - `apps/backend/interfaces/api/routers/tasks.py`
- Owns:
  - in-memory task snapshots
  - bounded replay buffer and gap detection
  - terminal `result|error|end` emission
  - cancellation API
- Open these files first when reconnect/replay/status drift is the seam.

<a id="map-owner-backend-logging"></a>
### Backend Logging seam
- Canonical owner: `apps/backend/runtime/logging.py`
- Owns:
  - normalized repo-owned logger acquisition via `get_backend_logger(...)`
  - logger-style repo-owned emission through the `BackendLoggerProxy`
  - plain human-readable wrapper emission via `emit_backend_message(...)`
  - structured telemetry emission via `emit_backend_event(...)`
  - canonical uvicorn/bootstrap logging config via `build_backend_uvicorn_log_config(...)`
- Secondary seams:
  - `apps/backend/interfaces/api/run_api.py`
  - `apps/backend/runtime/diagnostics/error_summary.py`
  - `apps/backend/runtime/diagnostics/exception_hook.py`
  - `apps/backend/infra/stdio.py`
- Open this when the issue is logger namespace normalization, repo-owned backend log formatting/emission, bootstrap/server log config, or the split between operational logs, concise runtime summaries, full exception dumps, and exact stdout/stderr contracts.

<a id="map-owner-orchestrator"></a>
### Orchestrator seam
- Canonical owner: `apps/backend/core/orchestrator.py`
- Owns:
  - engine registry lookup
  - load/unload/cache residency decisions
  - dispatch from task workers into engine wrappers
- It coordinates execution; it is not the terminal result owner.

<a id="map-owner-shared-pipeline-stages"></a>
### Shared pipeline stages
- Canonical owner: `apps/backend/runtime/pipeline_stages/`
- High-value entries for this atlas:
  - `hires_fix.py`
  - `masked_img2img.py`
  - `video.py`
  - `ip_adapter.py`
- Open this directory when the question is a shared stage reused by multiple canonical use-cases.
- `video.py` is the current shared owner for WAN Diffusers stage-LoRA preflight/apply across `txt2vid`, `img2vid`, and `vid2vid`.

## Generated artifact pointers
<a id="map-artifact-backend-header-snapshot"></a>
### Backend header snapshot
- Current generated owner: `.sangoi/reports/tooling/apps-backend-file-header-blocks.md`
- Use it when you need the current backend file-header snapshot that feeds the grouped book index.

<a id="map-artifact-backend-py-book-index"></a>
### Backend Python book index
- Current generated owner: `.sangoi/reports/tooling/backend-py-book-index.md`
- Use it when you need the full grouped backend file list and current generated subtree/timestamp scope.

## Regeneration pointers
- Root checklist for map/index refresh: `AGENTS.md`
- Tool command catalog: `.sangoi/.tools/AGENTS.md`
- Generated report catalog: `.sangoi/reports/tooling/AGENTS.md`
- Current backend discovery outputs:
  - `.sangoi/reports/tooling/apps-backend-file-header-blocks.md`
  - `.sangoi/reports/tooling/backend-py-book-index.md`

## Policy pointers
<a id="map-policy-root-agents"></a>
### Root AGENTS maintenance
- Canonical owner: `AGENTS.md`
- Use this when you need the same-tranche maintenance triggers for `SUBSYSTEM-MAP-INDEX.md`, `SUBSYSTEM-MAP.md`, and mapped file headers.

<a id="map-policy-file-headers"></a>
### File header standard
- Canonical owner: `.sangoi/policies/file-header-block.md`
- Use this when a touched `apps/**` file header needs truthful `Purpose` / `Symbols` sync.

<a id="map-policy-tooling-catalog"></a>
### Tooling catalog
- Canonical owner: `.sangoi/.tools/AGENTS.md`
- Use this when you need the current commands/failure modes for header dumps, backend book-index rebuilds, and link/header lint checks.

<a id="map-policy-report-catalogs"></a>
### Report catalogs
- Canonical owners:
  - `.sangoi/reports/AGENTS.md`
  - `.sangoi/reports/tooling/AGENTS.md`
- Use these when you need to know which generated reports are active report-catalog truth versus retained audit artifacts.
