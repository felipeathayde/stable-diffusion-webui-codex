# apps.launcher.gui_tk
Date: 2026-01-25
Status: Active
Last Review: 2026-02-23

## Purpose
- Modular Tk/ttk GUI implementation for the Codex launcher (services + settings + logs).
- Keep the stable entrypoint at `apps/codex_launcher.py` thin; implementation lives here.

## Modules
- `__init__.py` – Public re-exports (`CodexLauncherApp`, `main`).
- `app.py` – Tk root window + tab wiring + background task polling.
- `controller.py` – Non-UI controller (store/services/log buffer + persistence helpers).
- `styles.py` – Palette + ttk styling.
- `widgets.py` – Scrollable container + small layout helpers.
- `tabs/services.py` – API/UI supervision tab.
- `tabs/runtime.py` – Device defaults + attention mode + GGUF/LoRA + PyTorch alloc conf.
- `tabs/diagnostics.py` – Preflight checks + debug/logging env flags.
- `tabs/logs.py` – Log viewer (filter/search/export).

## Notes
- UI state (tab index, window geometry, external terminal) is auto-persisted via `LauncherProfileStore.save_meta()`; env changes require explicit “Save Settings”.
- Logs are structured (`CodexLogRecord`) and rendered incrementally to avoid UI freezes.
- 2026-01-30: Removed the dev-only Z-Image Diffusers bypass toggle (`CODEX_ZIMAGE_DIFFUSERS_BYPASS`) from `tabs/diagnostics.py`.
- 2026-01-31: `tabs/diagnostics.py` now exposes global profiling env flags (`CODEX_PROFILE*`) for backend torch-profiler runs.
- 2026-02-15: `tabs/diagnostics.py` now exposes launcher trace toggles (`CODEX_TRACE_CONTRACT`, `CODEX_TRACE_PROFILER`) alongside timeline/profile flags.
- 2026-02-18: `tabs/runtime.py` now exposes task cancel default mode (`CODEX_TASK_CANCEL_DEFAULT_MODE`) with strict choices (`immediate`, `after_current`) alongside existing task/safety knobs.
- 2026-02-21: `tabs/runtime.py` now exposes WAN img2vid chunk buffer mode (`CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE`) with strict choices (`hybrid`, `ram`, `ram+hd`).
- 2026-02-21: `tabs/runtime.py` now exposes attention mode selection (`sdpa_*|xformers|split|quad`) and persists it as launcher bootstrap env (`CODEX_ATTENTION_BACKEND`, `CODEX_ATTENTION_SDPA_POLICY`).
- 2026-02-22: `tabs/runtime.py` now uses `PYTORCH_ALLOC_CONF` for allocator tuning.
- 2026-02-22: `tabs/runtime.py` no longer offers GGUF dequant cache levels (`lvl1`/`lvl2`) or ratio tuning UI; the setting is locked to `off` and stale dequant cache env tuning keys are cleared during runtime sync.
- 2026-02-23: `tabs/runtime.py` now uses a single `CODEX_MAIN_DEVICE` selector and mirrors it to `CODEX_CORE_DEVICE`/`CODEX_TE_DEVICE`/`CODEX_VAE_DEVICE` to enforce main-device invariance from launcher bootstrap.
- 2026-02-23: `tabs/runtime.py::reload()` no longer mutates mount/offload to main-device as a side effect while loading UI state; it now parses and preserves persisted `CODEX_MOUNT_DEVICE`/`CODEX_OFFLOAD_DEVICE` values.
- 2026-02-23: `tabs/runtime.py::reload()` now defaults invalid/missing offload values to CPU (not main device) to keep launcher bootstrap aligned with explicit offload de-residency semantics.
