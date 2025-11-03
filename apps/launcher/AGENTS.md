# apps.launcher
Date: 2025-10-28
Owner: Repository Maintainers
Status: Active
Last Review: 2025-11-03

## Purpose
- Provide reusable launcher infrastructure (path resolution, environment checks, service supervision, segmented profile persistence) for Codex entrypoints.

## Modules
- `paths.py` – Resolve canonical data/model/output directories with strict normalisation.
- `log_buffer.py` – Thread-safe ring buffer for capturing launcher logs shared across services and UI.
- `checks.py` – Environment validation (Python, Node/npm, Vite) with explicit diagnostics.
- `services.py` – Service specifications and process supervision helpers for API/UI processes.
- `profiles.py` – Segmented profile persistence (`.sangoi/launcher/`) with area/model separation and legacy migration support.
- `__init__.py` – Re-exports public APIs for callers (`CodexPaths`, `run_launch_checks`, `LauncherProfileStore`, etc.).

## Notes
- New launcher features must surface through this package; avoid ad-hoc scripts accessing internal modules directly.
- Persistence writes to `.sangoi/launcher/{meta,areas,models}`; migrations should extend `profiles` rather than duplicating logic.
- Service command definitions should remain minimal and composition-friendly—prefer adding options via profile/env rather than bespoke subprocess code.
- 2025-11-02: Windows “Services in new terminal” now wraps commands with `cmd.exe /K` and leaves stdin attached so the console stays open after exit for manual inspection.
- 2025-11-02: Launcher profiles persist diffusion/TE/VAE device + dtype choices via the Codex options snapshot, and `services.py` now forwards them as CLI flags (`--core-device`, `--te-device`, `--vae-device`, etc.) when spawning the API instead of relying on env vars.
- 2025-11-03: Launcher forwards conditioning diagnostics via `--debug-conditioning` when `CODEX_DEBUG_COND` is enabled in profiles/TUI.
