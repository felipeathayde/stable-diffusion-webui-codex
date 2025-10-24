# Model Tabs and Workflows — Design Spec (MVP)

Status: Draft (2025-10-22)
Owner: WebUI Team
Scope: Frontend tabs (hand-crafted Vue views), minimal backend persistence + actions

## Goals
- Replace feature sprawl with first-class, “base tabs” per model family (SD15, SDXL, FLUX, WAN 2.2).
- Allow users to create multiple instances of a base tab (presets), rename, reorder, enable/disable, load/unload, and generate.
- For WAN 2.2, provide a single, configurable base that covers Normal and Lightning (LoRA per stage), with dual panels (High/Low) and a Video panel.
- Introduce a Workflows tab that receives immutable snapshots from base tabs via “Send to Workflows”, without exposing a free-form graph editor.
- Preserve QuickSettings for global options (dtype/memory/paths/vae/text encoder); gradually remove checkpoint/engine from header when tabs take ownership.

## Non-Goals (for MVP)
- No server-driven UI blocks in tabs (hand-crafted Vue only).
- No free-form workflow editor or JSON editing in Workflows (snapshots are read-only there).
- No auto-install of weights (buttons can exist but behave as stubs or simple validators in MVP).

## Definitions
- Base Tab: a typed, hand-crafted view (.vue) dedicated to a model family, with its own params and actions.
- Instance (aka Preset Tab): a user-created copy of a base tab with customized params and title.
- Snapshot (Workflow): an immutable capture of a tab’s params that can be run later from the Workflows tab.

## User Stories (abridged)
- “As a user, I want to create two WAN tabs, one Lightning 4-steps and one Normal, and switch between them quickly.”
- “As a user, I want to send the current settings from my WAN tab into Workflows and run it later without re-tuning.”
- “As a user, I want to load/unload a tab’s models to manage VRAM.”

## Principles
- Views are crafted manually (no block generators), to keep style and UX crisp.
- Tabs own model semantics and params; backend detects engine semantics automatically, not the UI.
- Snapshots are the only artifact the Workflows tab consumes; editing happens on source tabs only.

---

## Base Tabs (Types and Panels)

### Common Header (all tabs)
- Title (inline editable), Duplicate, Remove, Enable/Disable, Reorder (drag).
- Actions: Load / Unload (model(s) for this tab), Generate, Results.
- Status: Loaded/Unloaded, VRAM (optional), Task state.

### SD15Tab.vue / SDXLTab.vue / FLUXTab.vue
- Single “Generation Parameters” panel (prompt, negative, size, steps, sampler, scheduler, CFG, seed).
- Optional sub-panel (e.g., SDXL Refiner).
- Uses existing image generation endpoints.

### WANTab.vue (WAN 2.2)
- Switch “Lightning” (per-tab):
  - When ON, reveal LoRA controls per stage and suggest steps=4 for both stages.
- Panel “High Noise”
  - model_dir (path picker), sampler, scheduler, steps, cfg_scale, seed.
  - Switch “Use LoRA” → lora path + weight.
- Panel “Low Noise” (same fields).
- Panel “Video”
  - filename prefix, format, pix_fmt, CRF, loop_count, pingpong, save_metadata, save_output, fps, frames.
- Footer: Load/Unload (per tab), Generate, Results viewer.

Notes:
- The tab owns model paths (High/Low) and optional LoRAs; the header’s “Checkpoint” is hidden while in a WAN tab.
- Dual-stage execution already exists; this tab becomes the single control point.

---

## QuickSettings (MVP)
- Keep only global/infra options: dtype/memory budget, VAE, Text Encoder, paths.
- Hide “Checkpoint” and “Engine” when current tab manages models (e.g., WAN).

---

## Workflows Tab (Snapshots)
- Single list of workflows. Each item = immutable snapshot created via “Send to Workflows” from any base tab.
- Selecting a workflow shows a “Flow” panel with nodes derived from the snapshot type:
  - WAN snapshot → [HighStageNode, LowStageNode, VideoExportNode] with fixed connections.
  - SDXL snapshot → [DenoiseNode, (optional) RefinerNode].
- Actions: Rename workflow, Delete, “Load into base” (applies snapshot back to its source base type, opening/activating a tab). No in-place editing of the graph.

---

## State & Persistence
- Phase 1: localStorage (tab instances + workflows), with debounced saves.
- Phase 2: backend persistence in JSON files with mtime cache.
  - apps/interface/tabs.json, apps/interface/workflows.json

### tabs.json (v1)
{
  "version": 1,
  "tabs": [
    {
      "id": "uuid",
      "type": "wan",
      "title": "WAN Lightning 4 steps",
      "order": 1,
      "enabled": true,
      "installed": true,
      "params": {
        "wan": {
          "high": {
            "model_dir": "/models/Wan/Wan2.2-I2V-14B-High",
            "sampler": "Euler",
            "scheduler": "Simple",
            "steps": 4,
            "cfg_scale": 7.0,
            "seed": -1,
            "lightning": true,
            "lora": { "enabled": true, "path": "/models/loras/high_lora.safetensors", "weight": 1.0 }
          },
          "low": {
            "model_dir": "/models/Wan/Wan2.2-I2V-14B-Low",
            "sampler": "Euler",
            "scheduler": "Simple",
            "steps": 4,
            "cfg_scale": 5.0,
            "seed": -1,
            "lightning": true,
            "lora": { "enabled": true, "path": "/models/loras/low_lora.safetensors", "weight": 1.0 }
          },
          "video": {
            "filename_prefix": "wan22",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 15,
            "fps": 24,
            "frames": 16,
            "loop_count": 0,
            "pingpong": false,
            "save_metadata": true,
            "save_output": true
          }
        }
      },
      "meta": { "created_at": 173, "updated_at": 173 }
    }
  ]
}

### workflows.json (v1)
{
  "version": 1,
  "workflows": [
    {
      "id": "uuid",
      "name": "WAN Lightning 4 steps (1080p)",
      "type": "wan",
      "source_tab_type": "wan",
      "source_tab_title": "WAN Lightning 4 steps",
      "created_at": 173,
      "engine_semantics": "wan22",
      "params_snapshot": {},
      "nodes": [
        { "id": "high", "type": "HighStageNode", "params": {} },
        { "id": "low",  "type": "LowStageNode",  "params": {} },
        { "id": "mux",  "type": "VideoExportNode", "params": {} }
      ]
    }
  ]
}

---

## Backend API (Phase 2)

### Tabs
- GET /api/ui/tabs → { version, tabs }
- POST /api/ui/tabs → create or duplicate
  - Body: { type: 'wan'|'sdxl'|'flux'|'sd15', title?, params? } | { duplicate_of: id }
- PATCH /api/ui/tabs/:id → rename/update params/enable
  - Body: { title?, enabled?, params? }
- POST /api/ui/tabs/reorder → { ids: [id1, id2, ...] }
- DELETE /api/ui/tabs/:id
- POST /api/models/load → { tab_id } (loads models for tab), returns { loaded: true }
- POST /api/models/unload → { tab_id } (evicts models), returns { unloaded: true }

### Workflows
- GET /api/ui/workflows → { version, workflows }
- POST /api/ui/workflows → create snapshot
  - Body: { source_tab_id } (server copies params by id)
- PATCH /api/ui/workflows/:id → rename
- DELETE /api/ui/workflows/:id
- POST /api/ui/workflows/:id/load-into-base → returns a mutable params object suitable for the source base type

Errors: 400 (bad payload), 404 (not found), 409 (invalid state), 500 (internal).

---

## Frontend Architecture

### Router
- /models → list of tabs, “New Tab” (base type picker) and “Duplicate” actions.
- /models/:tabId → base tab view.
- /workflows → workflows list + read-only flow viewer.

### Stores
- useModelTabsStore
  - state: tabs[], activeId
  - actions: create(type), duplicate(id), remove(id), rename(id, title), reorder(ids), setEnabled(id, on), updateParams(id, partial), persist()
- useWorkflowsStore
  - state: workflows[]
  - actions: createFromTab(tabId), rename(id, name), remove(id), loadIntoBase(id)
- useTasksStore (existing): run jobs and stream results.

### Views & Components
- BaseTabHeader.vue (title, actions, status)
- WANTab.vue (Panels: High, Low, Video)
- SDXLTab.vue / FLUXTab.vue / SD15Tab.vue
- WorkflowsTab.vue (list + flow)
- Helpers: StagePanel.vue, VideoSettingsPanel.vue, ResultViewer (reuse), PromptFields (reuse)

---

## UX Flows
- Duplicate Tab: clones params+title ("Copy #n"), focuses the new route, persists order.
- Load/Unload: disabled while a task runs; shows persistent status in header.
- Send to Workflows: creates a snapshot; success toast; appears in Workflows list.
- Load into base: opens a new tab instance pre-populated from snapshot; user may rename and save.

---

## Testing & Acceptance (MVP)
- Tabs: CRUD, reorder, persistence across reloads.
- WAN: dual-stage (Normal & Lightning) end-to-end; LoRA toggles don’t break flow when off; export honors options (format/pix_fmt/CRF/ping-pong/loop).
- SDXL/FLUX/SD15: image generation parity with current views.
- Workflows: snapshot creation and round-trip via Load into base; no in-place editing.
- QuickSettings: globals work; checkpoint hidden where tabs manage model.

---

## Risks & Mitigations
- Conflicts between Load/Unload and Generate → disable/guard in UI + checks server-side.
- VRAM pressure (WAN high/low) → Unload button near the status; show VRAM total (optional).
- Snapshot drift → snapshots are immutable; editing always on tabs; “Load into base” clones params.

---

## Roadmap & Timeline (estimates)
- F0 Preparation: 1–2d
- F1 Front tabs infra (router+store+header): 3–4d
- F2 WANTab (High/Low/Video): 4–6d
- F3 SDXL/FLUX/SD15 tabs: 3–5d
- F4 Backend persistence (tabs/workflows CRUD + load/unload): 2–4d
- F5 Workflows tab (snapshots): 4–7d
- F6 Refinements (LoRA apply, GGUF stages, RIFE): ongoing

---

## Open Questions / Later
- Where to surface text encoder choice for WAN (UMT5-XXL): QuickSettings vs tab header.
- Multi-engine concurrency: should we allow multiple tabs “loaded” simultaneously; eviction policy.
- Export options per workflow: keep copy in snapshot; override per-run?

*** End of Spec ***
