# apps.launcher
Date: 2025-10-28
Status: Active
Last Review: 2026-02-23

## Purpose
- Provide reusable launcher infrastructure (path resolution, environment checks, service supervision, segmented profile persistence) for Codex entrypoints.

## Modules
- `paths.py` – Resolve canonical data/model/output directories with strict normalisation.
- `log_buffer.py` – Thread-safe ring buffer for capturing launcher logs shared across services and UI.
- `checks.py` – Environment validation (Python, Node/npm, Vite) with explicit diagnostics.
- `services.py` – Service specifications and process supervision helpers for API/UI processes.
- `profiles.py` – Segmented profile persistence (`.sangoi/launcher/`) with area/model separation and legacy migration support.
- `settings.py` – Typed launcher settings + validation helpers (env-backed, UI/service friendly).
- `gui_tk/` – Tk-based GUI launcher package (UI around profiles/checks/services).
- `__init__.py` – Re-exports public APIs for callers (`CodexPaths`, `run_launch_checks`, `LauncherProfileStore`, etc.).

## Notes
- New launcher features must surface through this package; avoid ad-hoc scripts accessing internal modules directly.
- Persistence writes to `.sangoi/launcher/{meta,areas,models}`; migrations should extend `profiles` rather than duplicating logic.
- Service command definitions should remain minimal and composition-friendly—prefer adding options via profile/env rather than bespoke subprocess code.
- The Tk GUI uses `logo.png` for `iconphoto` and (on Windows) `logo.ico` for `iconbitmap` so the taskbar icon is branded.
- Launcher meta (`.sangoi/launcher/meta.json`) stores `window_geometry` so the Tk GUI can restore size/position across runs.
- 2025-11-02: Windows “Services in new terminal” now wraps commands with `cmd.exe /K` and leaves stdin attached so the console stays open after exit for manual inspection.
- 2025-11-02: Launcher profiles persist diffusion/TE/VAE device + dtype choices via the Codex options snapshot, and `services.py` now forwards them as CLI flags (`--core-device`, `--te-device`, `--vae-device`, etc.) when spawning the API instead of relying on env vars.
- 2025-11-03: Launcher forwards conditioning diagnostics via `--debug-conditioning` when `CODEX_DEBUG_COND` is enabled in profiles/TUI.
- 2025-12-29: Launcher now resolves the repo root via `CODEX_ROOT` (shared helper) instead of `Path(__file__).parents[...]`, so Windows/WSL launch methods stay consistent.
- 2025-12-29: Launcher UI service now always receives `API_PORT` (prevents Vite proxy/API_PORT derivation from a fallback WEB_PORT), and the API service performs a strict preflight port check across IPv4/IPv6 localhost (helps diagnose WSL/Windows double-run and “localhost” split-brain).
- 2026-01-23: Launcher now persists GGUF/LoRA runtime knobs (`CODEX_GGUF_EXEC`, `CODEX_LORA_APPLY_MODE`, `CODEX_LORA_ONLINE_MATH`) and forwards them to the backend as CLI flags when starting the API (`--gguf-exec`, `--lora-apply-mode`, `--lora-online-math`). API is started via `apps/backend/interfaces/api/run_api.py` so backend args are supported.
- 2026-01-24: Launcher profiles include explicit device defaults (`CODEX_CORE_DEVICE`, `CODEX_TE_DEVICE`, `CODEX_VAE_DEVICE`) and `services.py` forwards them to the backend as CLI flags (`--core-device`, `--te-device`, `--vae-device`) to avoid bootstrap-time fallback/prompt failures in non-interactive spawns (and profile consistency keeps these keys, no accidental pruning).
- 2026-01-02: Added standardized file header docstrings to launcher modules (doc-only change; part of rollout).
- 2026-01-03: Added standardized file header docstrings to remaining launcher modules (`__init__.py`, `checks.py`, `log_buffer.py`, `paths.py`) (doc-only change; part of rollout).
- 2026-01-06: Launcher Python preflight now matches `.python-version` (3.12.10) instead of allowing stale 3.10/3.11.
- 2026-01-21: Launcher profiles now default `PYTORCH_CUDA_ALLOC_CONF` (global PyTorch CUDA allocator tuning) to `max_split_size_mb:256,garbage_collection_threshold:0.8` when unset.
- 2026-01-29: Launcher no longer exposes reserved `cuda_pack` GGUF exec mode; CodexPack packed GGUFs are auto-detected via `codex.pack.*` / `*.codexpack.gguf`. Legacy launcher configs are migrated to `dequant_forward`.
- 2026-01-31: Launcher profiles now persist global profiling env flags (`CODEX_PROFILE*`) and the GUI diagnostics tab exposes them for backend torch-profiler runs.
- 2026-02-15: Launcher API arg forwarding now includes trace toggles (`CODEX_TRACE_CONTRACT` -> `--trace-contract`, `CODEX_TRACE_PROFILER` -> `--trace-profiler`) for backend bootstrap alignment.
- 2026-02-18: Launcher task/runtime profile defaults now persist `CODEX_TASK_CANCEL_DEFAULT_MODE` (`immediate|after_current`) as a backend bootstrap knob for task cancel policy.
- 2026-02-21: Launcher profiles/settings now persist and validate `CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE` (`hybrid|ram|ram+hd`) as a runtime bootstrap knob used by WAN22 img2vid chunk buffering policy.
- 2026-02-21: Launcher Runtime now owns attention bootstrap policy via `CODEX_ATTENTION_BACKEND` + `CODEX_ATTENTION_SDPA_POLICY`, forwarding `--attention-backend` and `--attention-sdpa-policy` to backend startup.
- 2026-02-22: Launcher profiles now write `PYTORCH_ALLOC_CONF` (replacing deprecated `PYTORCH_CUDA_ALLOC_CONF`) for allocator tuning defaults.
- 2026-02-22: Removed GGUF dequant-forward run cache forwarding from launcher bootstrap args (`services.py` no longer emits `--gguf-dequant-cache*` flags); runtime env normalization now forces `CODEX_GGUF_DEQUANT_CACHE=off` and clears stale ratio/limit keys.
- 2026-02-23: Launcher now defines a global device authority via `CODEX_MAIN_DEVICE`; `services.py` forwards `--main-device` and mirrors core/TE/VAE flags to the same value to enforce single-device runtime invariant.
- 2026-02-23: `profiles.py` now treats `CODEX_*` runtime/device keys as area-scoped only (`core`): model overlays and non-core areas can no longer override `CODEX_MAIN_DEVICE`/`CODEX_MOUNT_DEVICE`/`CODEX_OFFLOAD_DEVICE` (prevents stale model JSON from defeating saved runtime-tab device settings).
- 2026-02-23: `run-webui.{bat,sh}` now strips legacy `PYTORCH_CUDA_ALLOC_CONF` from launcher entrypoint env and keeps only `PYTORCH_ALLOC_CONF` for runtime allocator configuration.
- 2026-02-23: launcher offload default is now explicit CPU: `services.py` forwards `--offload-device=cpu` when unset, and `profiles.py` defaults `CODEX_OFFLOAD_DEVICE=cpu` to avoid implicit same-device offload no-op states under Contract-R unload semantics.
- 2026-02-23: `services.py` now enforces `PYTORCH_ALLOC_CONF` allocator backend when `CODEX_CUDA_MALLOC=1` (requires/ensures `backend:cudaMallocAsync`, fail-loud on conflicting backend entries).
- 2026-02-23: `run-webui.sh` now makes `--cuda-malloc` / `CODEX_CUDA_MALLOC=1` effective by ensuring `PYTORCH_ALLOC_CONF` includes `backend:cudaMallocAsync` (and failing loud on invalid/conflicting allocator config).
