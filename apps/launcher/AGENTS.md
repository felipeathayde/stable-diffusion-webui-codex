# apps.launcher
Date: 2025-10-28
Owner: Repository Maintainers
Status: Active
Last Review: 2026-01-21

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
- 2025-12-29: Launcher now resolves the repo root via `CODEX_ROOT` (shared helper) instead of `Path(__file__).parents[...]`, so Windows/WSL launch methods stay consistent.
- 2025-12-29: Launcher UI service now always receives `API_PORT` (prevents Vite proxy/API_PORT derivation from a fallback WEB_PORT), and the API service performs a strict preflight port check across IPv4/IPv6 localhost (helps diagnose WSL/Windows double-run and “localhost” split-brain).
- 2026-01-23: Launcher now persists GGUF/LoRA runtime knobs (`CODEX_GGUF_EXEC`, `CODEX_LORA_APPLY_MODE`, `CODEX_LORA_ONLINE_MATH`) and forwards them to the backend as CLI flags when starting the API (`--gguf-exec`, `--lora-apply-mode`, `--lora-online-math`). API is started via `apps/backend/interfaces/api/run_api.py` so backend args are supported.
- 2026-01-02: Added standardized file header docstrings to launcher modules (doc-only change; part of rollout).
- 2026-01-03: Added standardized file header docstrings to remaining launcher modules (`__init__.py`, `checks.py`, `log_buffer.py`, `paths.py`) (doc-only change; part of rollout).
- 2026-01-06: Launcher Python preflight now matches `.python-version` (3.12.10) instead of allowing stale 3.10/3.11.
- 2026-01-21: Launcher profiles now default `PYTORCH_CUDA_ALLOC_CONF` (global PyTorch CUDA allocator tuning) to `max_split_size_mb:256,garbage_collection_threshold:0.8` when unset.
