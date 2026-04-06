<!-- tags: webui, architecture, map, index, discovery -->
# Subsystem Map Index
Date: 2026-03-31
Last Review: 2026-03-31
Status: Active

Read this first. This file is lookup-only.
Use `Jump to` to open the bounded section in `SUBSYSTEM-MAP.md`, then follow the canonical owner path into code or policy.

## Hotspots
| Type | Name | Aliases | Jump to | Canonical owner path |
| --- | --- | --- | --- | --- |
| Hotspot | Keymaps | checkpoint keyspace mapping; WAN LoRA logical keys; Anima raw `net.*` core keyspace | [Keymaps](SUBSYSTEM-MAP.md#map-hotspot-keymaps) | `apps/backend/runtime/state_dict/key_mapping.py` |
| Hotspot | Native Conv3D VAE | codex3d vae; conv3d native | [Native Conv3D VAE](SUBSYSTEM-MAP.md#map-hotspot-vae-codex3d) | `apps/backend/runtime/common/vae_codex3d.py` |
| Hotspot | Hires Fix | hires second pass; high-res fix | [Hires Fix](SUBSYSTEM-MAP.md#map-hotspot-hires-fix) | `apps/backend/runtime/pipeline_stages/hires_fix.py` |
| Hotspot | Image Automation | infinite generate; folder loop; wildcard batch | [Image Automation](SUBSYSTEM-MAP.md#map-hotspot-image-automation) | `apps/backend/use_cases/image_automation.py` |
| Hotspot | IP-Adapter | image prompt adapter; reference image conditioning | [IP-Adapter](SUBSYSTEM-MAP.md#map-hotspot-ip-adapter) | `apps/backend/runtime/pipeline_stages/ip_adapter.py` |
| Hotspot | SUPIR Runtime | supir mode; sdxl restore; img2img restore branch | [SUPIR Runtime](SUBSYSTEM-MAP.md#map-hotspot-supir-runtime) | `apps/backend/runtime/families/supir/runtime.py` |
| Hotspot | attention_sram_v1 | SRAM attention; split-KV; shared-memory attention | [attention_sram_v1](SUBSYSTEM-MAP.md#map-hotspot-attention-sram-v1) | `apps/backend/runtime/attention/sram/__init__.py` |
| Hotspot | Generation Router | generation endpoints; request parsing; SUPIR preflight | [Generation Router](SUBSYSTEM-MAP.md#map-hotspot-generation-router) | `apps/backend/interfaces/api/routers/generation.py` |
| Hotspot | Task Streaming | task SSE; replay buffer; `/api/tasks/{id}/events` | [Task Streaming](SUBSYSTEM-MAP.md#map-hotspot-task-streaming) | `apps/backend/interfaces/api/routers/tasks.py` |
| Hotspot | Orchestrator | engine dispatch; model load/cache; runtime coordinator | [Orchestrator](SUBSYSTEM-MAP.md#map-hotspot-orchestrator) | `apps/backend/core/orchestrator.py` |

## Pipelines
| Type | Name | Aliases | Jump to | Canonical owner path |
| --- | --- | --- | --- | --- |
| Pipeline | txt2img | text-to-image; image gen | [txt2img](SUBSYSTEM-MAP.md#map-pipeline-txt2img) | `apps/backend/use_cases/txt2img.py` |
| Pipeline | img2img | image-to-image; inpaint; hires continuation; SUPIR mode; mask-capability gate | [img2img](SUBSYSTEM-MAP.md#map-pipeline-img2img) | `apps/backend/use_cases/img2img.py` |
| Pipeline | image_automation | infinite generate; repeat loop; batch method | [image_automation](SUBSYSTEM-MAP.md#map-pipeline-image-automation) | `apps/backend/use_cases/image_automation.py` |
| Pipeline | txt2vid | text-to-video | [txt2vid](SUBSYSTEM-MAP.md#map-pipeline-txt2vid) | `apps/backend/use_cases/txt2vid.py` |
| Pipeline | img2vid | image-to-video | [img2vid](SUBSYSTEM-MAP.md#map-pipeline-img2vid) | `apps/backend/use_cases/img2vid.py` |
| Pipeline | vid2vid | video-to-video; parked public route; WAN animate dormant lane | [vid2vid](SUBSYSTEM-MAP.md#map-pipeline-vid2vid) | `apps/backend/use_cases/vid2vid.py` |
| Pipeline | API bootstrap | FastAPI mount; router registration; shared bootstrap logging handoff | [API bootstrap](SUBSYSTEM-MAP.md#map-pipeline-api-bootstrap) | `apps/backend/interfaces/api/run_api.py` |

## Owner Seams
| Type | Name | Aliases | Jump to | Canonical owner path |
| --- | --- | --- | --- | --- |
| Owner Seam | Generation Router | route contract; task spawn; payload DTO parse | [Generation Router seam](SUBSYSTEM-MAP.md#map-owner-generation-router) | `apps/backend/interfaces/api/routers/generation.py` |
| Owner Seam | Image Task Worker | image task lifecycle; result packaging | [Image Task Worker](SUBSYSTEM-MAP.md#map-owner-image-task-worker) | `apps/backend/interfaces/api/tasks/generation_tasks.py` |
| Owner Seam | Task Registry + SSE | task snapshot; replay buffer; cancel/end | [Task Registry + SSE](SUBSYSTEM-MAP.md#map-owner-task-registry-sse) | `apps/backend/interfaces/api/task_registry.py` |
| Owner Seam | Backend Logging | logger wrapper; BackendLoggerProxy; bootstrap log config; repo-owned emission path | [Backend Logging seam](SUBSYSTEM-MAP.md#map-owner-backend-logging) | `apps/backend/runtime/logging.py` |
| Owner Seam | Orchestrator | engine load/cache; dispatch to wrappers | [Orchestrator seam](SUBSYSTEM-MAP.md#map-owner-orchestrator) | `apps/backend/core/orchestrator.py` |
| Owner Seam | Shared Pipeline Stages | prompt; sampling; hires; video; adapters | [Shared pipeline stages](SUBSYSTEM-MAP.md#map-owner-shared-pipeline-stages) | `apps/backend/runtime/pipeline_stages/` |

## Generated Artifacts
| Type | Name | Aliases | Jump to | Canonical owner path |
| --- | --- | --- | --- | --- |
| Generated Artifact | Backend header snapshot | backend file-header snapshot; backend header dump | [Backend header snapshot](SUBSYSTEM-MAP.md#map-artifact-backend-header-snapshot) | `.sangoi/reports/tooling/apps-backend-file-header-blocks.md` |
| Generated Artifact | Backend Python book index | backend grouped index; backend book | [Backend Python book index](SUBSYSTEM-MAP.md#map-artifact-backend-py-book-index) | `.sangoi/reports/tooling/backend-py-book-index.md` |

## Policy Pointers
| Type | Name | Aliases | Jump to | Canonical owner path |
| --- | --- | --- | --- | --- |
| Policy Pointer | Root AGENTS maintenance rules | map maintenance; same-tranche sync | [Root AGENTS maintenance](SUBSYSTEM-MAP.md#map-policy-root-agents) | `AGENTS.md` |
| Policy Pointer | File header standard | header policy; Purpose/Symbols standard | [File header standard](SUBSYSTEM-MAP.md#map-policy-file-headers) | `.sangoi/policies/file-header-block.md` |
| Policy Pointer | Tooling catalog | doc tooling; generator commands | [Tooling catalog](SUBSYSTEM-MAP.md#map-policy-tooling-catalog) | `.sangoi/.tools/AGENTS.md` |
| Policy Pointer | Report catalogs | report owners; tooling report catalog | [Report catalogs](SUBSYSTEM-MAP.md#map-policy-report-catalogs) | `.sangoi/reports/AGENTS.md` |
