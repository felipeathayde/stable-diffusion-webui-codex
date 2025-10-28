# apps Overview
Date: 2025-10-28
Owner: Repository Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Host all active application code for the Codex rebuild. Each top-level package under `apps/` owns a distinct runtime surface (backend services, new Vue interface, launcher tooling).

## Subdirectories
- `backend/` — Codex-native backend stack (engines, runtimes, services, registries, HF asset mirrors). This is the authoritative implementation.
- `interface/` — Vue 3 + Vite application that replaces the legacy Gradio UI. Includes build tooling, public assets, and source modules.

## Key Files
- `launcher.py` — Process launcher used by CLI/TUI tooling to start backend and interface services.
- `__init__.py` — Marks `apps` as a Python package so relative imports resolve cleanly across backend modules.

## Notes
- New code must target `apps/backend` and `apps/interface`. The launcher lives at `apps/launcher.py`.
- When adding new subpackages, create an `AGENTS.md` describing responsibilities to keep this overview accurate.
