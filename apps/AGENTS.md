# apps Overview
Date: 2025-10-28
Owner: Repository Maintainers
Last Review: 2026-01-03
Status: Active

## Purpose
- Host all active application code for the Codex rebuild. Each top-level package under `apps/` owns a distinct runtime surface (backend services, new Vue interface, launcher tooling).

## Subdirectories
- `backend/` — Codex-native backend stack (engines, runtimes, services, registries, HF asset mirrors). This is the authoritative implementation.
- `interface/` — Vue 3 + Vite application that replaces the legacy Gradio UI. Includes build tooling, public assets, and source modules.

## Key Files
- `launcher/` — Package exposing launcher infrastructure (`checks`, `services`, `profiles`, `paths`).
- `tui_launcher.py` — Curses-based TUI entrypoint that drives the launcher package.
- `codex_launcher.py` — Tk-based GUI launcher for managing API/UI services (Windows).
- `__init__.py` — Marks `apps` as a Python package so relative imports resolve cleanly across backend modules.

## Notes
- New code must target `apps/backend` and `apps/interface`. The launcher lives under `apps/launcher/` with the TUI at `apps/tui_launcher.py`.
- When adding new subpackages, create an `AGENTS.md` describing responsibilities to keep this overview accurate.
- 2025-11-02: Launcher/TUI surfaces device/dtype configuration via persisted WebUI options (`apps/settings_values.json`); env overrides for runtime settings were removed.
- 2025-11-02: launcher services persist selections in `apps/settings_values.json`; when unset, backend startup defaults components to CPU with a warning.
- 2025-11-03: TUI runtime settings (device/dtype/attention/cache/offload) are configured via Web UI; launchers no longer write CODEX_* runtime settings env vars.
- 2025-11-03: BIOS TUI exposes "Conditioning Debug" toggle, wiring to `CODEX_DEBUG_COND` and the backend `--debug-conditioning` flag.
- 2025-11-03: (Deprecated) "Pin Shared Memory" launcher toggle removed; `--pin-shared-memory` remains a CLI-only switch (no env overrides).
- 2025-11-03: Logging tab now includes a "Trace Debug" toggle that sets `CODEX_TRACE_DEBUG=1`, enabling the global call tracer behind `--trace-debug`.
- 2025-11-25: BIOS Logging/Debug tab gained sampler diagnostics toggles: `CODEX_LOG_SAMPLER` (per-step norms) and `CODEX_LOG_SIGMAS` (sigma ladder dump).
- 2026-01-02: Launchers now expose `CODEX_LOG_CFG_DELTA` (and `CODEX_LOG_CFG_DELTA_N`) to log the cond/uncond delta inside CFG for the first N steps (requires `CODEX_LOG_SAMPLER=1`).
- 2025-11-14: BIOS DEBUG tab mirrors backend defaults for `CODEX_TRACE_DEBUG_MAX_PER_FUNC` (10 by default) so the displayed values stay in sync with `apps.backend.infra.config.args`.
- 2025-12-03: (Deprecated) "Force Native Sampler" launcher toggle removed; sampler routing is configured via Web UI / payload (no env overrides).
- 2025-12-28: `apps/settings_values.json` and `apps/interface/{tabs,workflows}.json` are backend-managed runtime state files; they are created/overwritten locally and are intentionally ignored by Git.
- 2025-12-29: Repo-root resolution across backend + launchers is now strict and `CODEX_ROOT`-anchored (no `__file__`/CWD fallbacks); launch via `run-webui.{bat,sh}` or set `CODEX_ROOT` explicitly.
- 2026-01-03: Added standardized file header docstrings to the remaining low-core `apps/` entrypoints (`__init__.py`, `backend/__init__.py`, and WebUI entrypoints/config) (doc-only change; part of rollout).
