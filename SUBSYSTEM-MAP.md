<!-- tags: webui, architecture, map, discovery, backend -->
# WebUI Subsystem Map (Macro Discovery)
Date: 2026-02-16
Last Review: 2026-02-16
Version: 2026-02-16
Status: Active

This document is a **discovery map**: `concept -> canonical location`.
Use it to find where things live fast, then open the file directly.

Contract-heavy policy details are intentionally kept out of this file.
See `AGENTS.md` and `.sangoi/reference/**` for authoritative contracts.

## Ownership Boundary (Manual vs Generated)
- Manual map (this file): discovery navigation, concept anchors, and macro subsystem orientation.
- Generated artifacts:
  - `.sangoi/reports/tooling/apps-backend-file-header-blocks.md`
  - `.sangoi/reports/tooling/backend-py-book-index.md`
- Rule: edit this file manually; regenerate the tooling artifacts with their commands, do not hand-edit generated outputs.

## Quick Start
1. Find your concept in **Fast Lookup**.
2. Jump to the anchor for canonical paths and secondary ownership paths.
3. If you need the full backend index, open `.sangoi/reports/tooling/backend-py-book-index.md`.

## Fast Lookup (Concept -> Where)
| Concept | Aliases | Go to |
|---|---|---|
| Keymaps | checkpoint key mapping, VAE key remap, TE key remap | [Keymaps](#concept-keymaps) |
| Native Conv3D VAE | conv3d native, codex3d vae | [Native Conv3D VAE](#concept-conv3d-vae) |
| Hires Fix | hires second pass, high-res fix | [Hires Fix](#concept-hires-fix) |
| Mode Pipelines | txt2img/img2img/txt2vid/img2vid/vid2vid flow | [Mode Pipelines](#concept-mode-pipelines) |
| Generation Router | generation endpoints, request normalization | [Generation Router](#concept-generation-router) |
| Task SSE | task events stream, `/api/tasks/{id}/events` | [Task Streaming](#concept-task-streaming) |
| Task Worker | async generation task execution | [Generation Task Worker](#concept-generation-task-worker) |
| Orchestrator | engine dispatch/load/cache | [Orchestrator](#concept-orchestrator) |
| Model Loader | checkpoint/diffusers loader | [Model Loader](#concept-model-loader) |
| Sampling Core | samplers, scheduler dispatch | [Sampling](#concept-sampling) |
| Memory + Smart Offload | VRAM policy, staged offload | [Memory](#concept-memory) |
| ControlNet Runtime | controlnet runtime/patch attach | [ControlNet](#concept-controlnet) |
| GGUF Converter | safetensors -> gguf tool | [GGUF Converter](#concept-gguf-converter) |
| GGUF Runtime IO | gguf read/write/load | [GGUF IO](#concept-gguf-io) |
| Engine Spec + Factory | family assembly seams | [Engine Assembly](#concept-engine-assembly) |
| API Entrypoint | FastAPI bootstrap | [API Entrypoint](#concept-api-entrypoint) |
| Frontend Payload Builders | request builders for image/video | [Frontend Builders](#concept-frontend-builders) |
| Model Paths Roots | paths registry for scanners/UI filtering | [Model Roots](#concept-model-roots) |
| Full Backend Index | 492 tracked backend `.py` files | [Backend Book Index](#backend-book-index) |

## Ambiguous Terms (Disambiguation)
- `keymaps`:
  - Primary ownership: `apps/backend/runtime/state_dict/key_mapping.py`, `apps/backend/runtime/state_dict/keymap_wan22_transformer.py`
  - Secondary ownership: `apps/backend/quantization/codexpack_keymaps.py`, `apps/backend/runtime/tools/gguf_converter_key_mapping.py`
- `hires fix`:
  - Primary ownership: `apps/backend/runtime/pipeline_stages/hires_fix.py`
  - Secondary family implementation: `apps/backend/runtime/families/sd/hires_fix.py`
- `conv3d native`:
  - Primary ownership: `apps/backend/runtime/common/vae_codex3d.py`
  - Secondary lane integration: `apps/backend/runtime/families/wan22/vae_io.py`

## Backend Topology Snapshot (Tracked `.py`)
Snapshot date: 2026-02-16

- Total tracked backend Python files: **492**
- Canonical source for full grouped listing: `.sangoi/reports/tooling/backend-py-book-index.md`
- Header snapshot source: `.sangoi/reports/tooling/apps-backend-file-header-blocks.md`

| Subtree | `.py` files |
|---|---:|
| `runtime` | 285 |
| `engines` | 48 |
| `patchers` | 32 |
| `interfaces` | 31 |
| `infra` | 22 |
| `quantization` | 17 |
| `core` | 15 |
| `use_cases` | 12 |
| `video` | 9 |
| `inventory` | 8 |
| `services` | 6 |
| `types` | 4 |
| `huggingface` | 2 |
| `(root)` | 1 |

## Concept Anchors

<a id="concept-keymaps"></a>
### Keymaps
- Canonical: `apps/backend/runtime/state_dict/key_mapping.py`
- Canonical: `apps/backend/runtime/state_dict/keymap_wan22_transformer.py`
- Secondary: `apps/backend/quantization/codexpack_keymaps.py`
- Secondary: `apps/backend/runtime/tools/gguf_converter_key_mapping.py`
- Why: key-style detection/remap ownership for checkpoint, VAE, and text-encoder layouts.

<a id="concept-conv3d-vae"></a>
### Native Conv3D VAE
- Canonical: `apps/backend/runtime/common/vae_codex3d.py`
- Secondary: `apps/backend/runtime/families/wan22/vae_io.py`
- Why: native 3D VAE lane and WAN22 VAE lane selection/loading seam.

<a id="concept-hires-fix"></a>
### Hires Fix
- Canonical: `apps/backend/runtime/pipeline_stages/hires_fix.py`
- Secondary: `apps/backend/runtime/families/sd/hires_fix.py`
- Call sites: `apps/backend/use_cases/img2img.py`, `apps/backend/use_cases/txt2img_pipeline/runner.py`
- Why: shared hires stage with SD-family delegation.

<a id="concept-mode-pipelines"></a>
### Mode Pipelines
- Canonical: `apps/backend/use_cases/txt2img.py`
- Canonical: `apps/backend/use_cases/img2img.py`
- Canonical: `apps/backend/use_cases/txt2vid.py`
- Canonical: `apps/backend/use_cases/img2vid.py`
- Canonical: `apps/backend/use_cases/vid2vid.py`
- Why: canonical per-mode pipeline ownership.

<a id="concept-generation-router"></a>
### Generation Router
- Canonical: `apps/backend/interfaces/api/routers/generation.py`
- Why: request contract validation + dispatch entrypoints.

<a id="concept-task-streaming"></a>
### Task Streaming (SSE)
- Canonical: `apps/backend/interfaces/api/routers/tasks.py`
- Secondary: `apps/backend/interfaces/api/task_registry.py`
- Why: task status/events endpoint + replay buffering ownership.

<a id="concept-generation-task-worker"></a>
### Generation Task Worker
- Canonical: `apps/backend/interfaces/api/tasks/generation_tasks.py`
- Why: async task lifecycle, progress/result/error emission.

<a id="concept-orchestrator"></a>
### Orchestrator
- Canonical: `apps/backend/core/orchestrator.py`
- Why: engine resolution/load/cache/purge/dispatch coordination.

<a id="concept-model-loader"></a>
### Model Loader
- Canonical: `apps/backend/runtime/models/loader.py`
- Secondary: `apps/backend/runtime/model_registry/loader.py`
- Why: checkpoint/model assembly + model signature detection flow.

<a id="concept-sampling"></a>
### Sampling
- Canonical: `apps/backend/runtime/sampling/driver.py`
- Secondary: `apps/backend/runtime/sampling/catalog.py`
- Secondary: `apps/backend/runtime/sampling/registry.py`
- Why: native sampling execution and sampler/scheduler catalogs.

<a id="concept-memory"></a>
### Memory + Smart Offload
- Canonical: `apps/backend/runtime/memory/manager.py`
- Secondary: `apps/backend/runtime/memory/smart_offload.py`
- Secondary: `apps/backend/runtime/memory/smart_offload_invariants.py`
- Why: memory policy, loaded-model registry, offload invariants.

<a id="concept-controlnet"></a>
### ControlNet
- Canonical: `apps/backend/runtime/controlnet/runtime.py`
- Secondary: `apps/backend/patchers/controlnet/apply.py`
- Why: runtime wrapper and graph attach patching seam.

<a id="concept-gguf-converter"></a>
### GGUF Converter
- Canonical: `apps/backend/runtime/tools/gguf_converter.py`
- Secondary: `apps/backend/runtime/tools/gguf_converter_profiles.py`
- Why: deterministic SafeTensors -> GGUF conversion toolchain.

<a id="concept-gguf-io"></a>
### GGUF IO
- Canonical: `apps/backend/quantization/gguf/reader.py`
- Canonical: `apps/backend/quantization/gguf/writer.py`
- Secondary: `apps/backend/quantization/gguf_loader.py`
- Why: GGUF read/write/load seams used by tooling/runtime.

<a id="concept-engine-assembly"></a>
### Engine Assembly (spec + factory)
- Pattern: `apps/backend/engines/<family>/spec.py`
- Pattern: `apps/backend/engines/<family>/factory.py`
- Why: runtime container specification + family assembly boundary.

<a id="concept-api-entrypoint"></a>
### API Entrypoint
- Canonical: `apps/backend/interfaces/api/run_api.py`
- Why: FastAPI composition and router wiring.

<a id="concept-frontend-builders"></a>
### Frontend Payload Builders
- Canonical: `apps/interface/src/composables/useGeneration.ts`
- Canonical: `apps/interface/src/composables/useVideoGeneration.ts`
- Secondary: `apps/interface/src/stores/quicksettings.ts`
- Why: request payload builders and selector/sha resolution on UI side.

<a id="concept-model-roots"></a>
### Model Roots
- Canonical: `apps/paths.json`
- Secondary: `apps/backend/interfaces/api/routers/paths.py`
- Why: model root registry consumed by scanners and UI filtering.

<a id="backend-book-index"></a>
## Backend Book Index
- Full grouped backend list (492 tracked `.py`): `.sangoi/reports/tooling/backend-py-book-index.md`
- Backend header snapshot (Purpose/Symbols source): `.sangoi/reports/tooling/apps-backend-file-header-blocks.md`
- Ownership boundary:
  - Generated docs are regenerated via tools.
  - This map is the manual navigation layer that links those artifacts.

Regeneration:
1. `backend_py_paths_file="$(mktemp /tmp/backend_py_paths.XXXXXX.txt)"`
2. `git ls-files apps/backend | rg "\\.py$" | LC_ALL=C sort > "$backend_py_paths_file"`
3. `python3 .sangoi/.tools/dump_apps_file_headers.py --out .sangoi/reports/tooling/apps-backend-file-header-blocks.md --root apps/backend --fail-on-missing`
4. `python3 .sangoi/.tools/build_backend_py_book_index.py --paths "$backend_py_paths_file" --headers .sangoi/reports/tooling/apps-backend-file-header-blocks.md --out .sangoi/reports/tooling/backend-py-book-index.md`
5. `python3 .sangoi/.tools/build_backend_py_book_index.py --paths "$backend_py_paths_file" --headers .sangoi/reports/tooling/apps-backend-file-header-blocks.md --out .sangoi/reports/tooling/backend-py-book-index.md --check`

## Contract / Policy Pointers
- Root governance and ownership rules: `AGENTS.md`
- Model assets contract: `.sangoi/reference/models/model-assets-selection-and-inventory.md`
- API task/SSE contract: `.sangoi/reference/api/tasks-and-streaming.md`
- UI model tabs and generation flow: `.sangoi/reference/ui/model-tabs-and-quicksettings.md`
- Python facade policy: `.sangoi/policies/python-package-facades.md`
