# apps/backend/codex Overview
Date: 2025-10-28
Owner: Backend Maintainers
Last Review: 2025-12-29
Status: Transitional

## Purpose
- Provides Codex-specific bootstrap helpers retained from the migration phase (initialization hooks, options loader, LoRA selection state) used while upstream callers transition to the new backend APIs.

## Key Files
- `initialization.py` — Entry points executed during CLI/TUI startup.
- `loader.py` — Bundle-aware loader helpers for compatibility layers; instantiates engines via the registry.
- `options.py` — Accessors bridging to the native options store during the transition.
- `lora.py` — Shared LoRA selection state consumed by legacy surfaces.
- `main.py` — Historical CLI entry helper retained for reference.

## Notes
- New features should target the native services/options stack; keep this package stable until all legacy paths are retired. Loader helpers now surface `DiffusionModelBundle` objects and delegate to the engine registry.
- Once the last compatibility consumers migrate, archive this package under `.sangoi/deprecated/` and update the documentation.
- 2025-11-02: `options.py` now persists core/TE/VAE device + dtype selections (`codex_diffusion_device`, `codex_te_device`, `codex_vae_device`, etc.) so launcher/bootstrap code can reconstruct CLI overrides without env vars.
- 2025-12-05: Options snapshot/include flags `codex_smart_offload`/`codex_smart_fallback`, persisted in `apps/settings_values.json` and surfaced via `/api/options` so frontend quicksettings can drive smart offload/fallback behaviour without process restarts.
- 2025-12-29: `options.py` now resolves `apps/settings_values.json` relative to `CODEX_ROOT` (required) so option persistence doesn’t depend on the process CWD.
