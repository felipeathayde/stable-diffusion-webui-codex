# Apps Structure & Functionality Improvements
Date: 2025-10-28
Owner: Repository Maintainers
Last Review: 2025-10-28
Status: Draft

## 1. Consolidate Launcher Tooling
- Move `apps/launcher.py` into a namespace package (`apps/launcher/__init__.py`, `cli.py`).
- Add entrypoint helpers so tools/TUI use the same import surface.
- Update `tools/tui_bios.py` and future scripts to import from the new package.

## 2. Retire Transitional Modules
- Audit `apps/backend/codex` and `apps/backend/runtime/modules/` for active code.
  - For useful logic, port into the appropriate runtime/service modules using Codex conventions.
  - Archive or delete leftover shims once migration is complete.

## 3. Portal & Snapshot Documentation
- Create a top-level index (`.sangoi/AGENTS.md` or similar) linking to major directories.
- Generate a static snapshot of the `apps/` structure (leveraging existing `AGENTS.md`) and store under `apps/structure-overview.md` or similar.

## 4. Update Messages Referencing Old Launcher Paths
- Track down error/help strings in `modules/*` or elsewhere that still mention `apps.server.launcher`.
- Link each message to the appropriate NH task from the parity plan and update references to `apps.launcher` when the task executes.

## 5. Runtime/Engine Modularization
- Evaluate WAN22 runtime (`apps/backend/runtime/wan22/`) for further stage-level submodules (conditioning, scheduler, export).
- Assess existing `AGENTS.md` mapping to spot opportunities for new `use_cases/*` when combining features (e.g., ControlNet + LoRA flows).

## 6. Tooling Notes
- `tools/` scripts are internal helpers; no change required, but keep documentation accurate as structure evolves.
