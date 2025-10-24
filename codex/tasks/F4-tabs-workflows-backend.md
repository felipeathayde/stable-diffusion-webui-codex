# F4 — Tabs & Workflows Backend Persistence + Load/Unload

Status: Planned (2025-10-22)
Owner: WebUI Team
Depends on: F1 (tabs infra), F2/F3 (views), design spec codex/design/model-tabs-and-workflows.md
Scope: Backend (FastAPI) + light frontend wiring (persist/hydrate, actions).

## Summary
Implement backend JSON persistence for tabs and workflows with mtime caching and CRUD endpoints. Add model Load/Unload actions per tab (orchestrator load/evict). Keep UI mostly unchanged; only wire persistence and actions.

## Deliverables
- JSON stores with mtime cache:
  - `apps/ui/tabs.json` — registry of base tab instances.
  - `apps/ui/workflows.json` — registry of workflow snapshots.
- Endpoints (FastAPI):
  - Tabs: GET/POST/PATCH/DELETE + reorder.
  - Workflows: GET/POST/PATCH/DELETE + load-into-base.
  - Models: POST /api/models/load, POST /api/models/unload (by tab_id).
- Validation + error handling; versioned payloads; atomic writes.
- Frontend: hydrate/persist via store; wire Load/Unload buttons to backend.

## Non-Goals
- No schema migrations tooling (manual minimal migration only).
- No remote downloads of weights (local-only in MVP).

---

## Backend Tasks

### 1) Storage & Cache
- [ ] Files: `apps/ui/tabs.json`, `apps/ui/workflows.json`.
- [ ] Helpers: `_load_json(path)`, `_save_json(path, data)`, mtime cache same as /api/ui/blocks.
- [ ] Schema version: `{ version: 1, tabs: [] }`, `{ version: 1, workflows: [] }`.

### 2) Tabs Endpoints
- [ ] GET `/api/ui/tabs` → `{ version, tabs }` (sorted by `order`).
- [ ] POST `/api/ui/tabs`:
  - [ ] Create: `{ type, title?, params? }` → returns `{ id }`.
  - [ ] Duplicate: `{ duplicate_of }` → deep copy; adjust title ("Copy #n").
- [ ] PATCH `/api/ui/tabs/:id` → partial update `{ title?, enabled?, params? }`.
- [ ] POST `/api/ui/tabs/reorder` → `{ ids: [...] }`.
- [ ] DELETE `/api/ui/tabs/:id`.
- [ ] Guards: 404 if not found; 409 if invalid state (e.g., running task on delete); 400 for payload.

### 3) Workflows Endpoints
- [ ] GET `/api/ui/workflows` → `{ version, workflows }`.
- [ ] POST `/api/ui/workflows` → `{ source_tab_id }` → snapshot from tabs.json.
- [ ] PATCH `/api/ui/workflows/:id` → rename.
- [ ] DELETE `/api/ui/workflows/:id`.
- [ ] POST `/api/ui/workflows/:id/load-into-base` → returns `{ type, params }` (server-side validation of type).

### 4) Models Actions
- [ ] POST `/api/models/load` → `{ tab_id }`:
  - [ ] Resolve tab; depending on `type`, load appropriate engine resources.
  - [ ] For WAN: if params.wan.high/low.model_dir present → preload stage pipelines via loader.
- [ ] POST `/api/models/unload` → `{ tab_id }`:
  - [ ] Evict engine(s) for the tab (orchestrator.evict for key(s)).
- [ ] Return `{ loaded: true }` / `{ unloaded: true }` and current state summary.
- [ ] Guards: 409 if a task is running for that engine; 404 tab not found.

### 5) Concurrency & Safety
- [ ] Serialize writes to JSON files (simple file lock or in-process lock).
- [ ] Reject destructive ops while a related task runs (best-effort check via task registry).
- [ ] Normalize paths (absolute/relative) and reject parent traversal.

### 6) Logging & Errors
- [ ] Each endpoint logs intent and result.
- [ ] Consistent error payloads: `{ error, detail?, code }`.

---

## Frontend Wiring

### 7) Hydration/Persistence
- [ ] Add `tabsHydrate()` and `tabsPersist()` to `useModelTabsStore` with feature flag to switch localStorage → backend.
- [ ] Workflows store (new) hydrates/persists via backend endpoints.

### 8) Actions
- [ ] Header buttons invoke POST `/api/models/load|unload` and disable while running.
- [ ] TabsList: create/duplicate/delete/reorder via backend.
- [ ] Workflows: "Send to Workflows" calls POST `/api/ui/workflows` with source_tab_id.

### 9) Gating
- [ ] Hide QuickSettings "Checkpoint" for WAN tab (as in F2), independent of backend.

---

## QA
- [ ] CRUD Tabs persists to `apps/ui/tabs.json` and reflects in UI after reload.
- [ ] CRUD Workflows persists to `apps/ui/workflows.json`.
- [ ] Load/Unload returns success and UI status changes accordingly; Unload blocked during active task.
- [ ] Error cases: bad ids (404), bad payload (400), conflict (409) → proper messages.

## Security
- [ ] Validate and sanitize file paths (deny traversal).
- [ ] Do not allow arbitrary command execution; no downloads in MVP.

## Timeline (F4)
- Backend endpoints + storage: 1.5–2.5d
- Front wiring + QA: 1–2d

## Acceptance Criteria
- Tabs/workflows CRUD round-trip with backend JSON persistence.
- Load/Unload works for WAN (preload high/low when configured) and does not conflict with running jobs.
- UI integrates actions without regressions.
