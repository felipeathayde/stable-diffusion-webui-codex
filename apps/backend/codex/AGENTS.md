# apps/backend/codex Overview
Date: 2025-10-28
Owner: Backend Maintainers
Last Review: 2025-10-28
Status: Transitional

## Purpose
- Provides Codex-specific bootstrap helpers retained from the migration phase (initialization hooks, options loader, LoRA selection state) used while upstream callers transition to the new backend APIs.

## Key Files
- `initialization.py` — Entry points executed during CLI/TUI startup.
- `loader.py` — Legacy loader helpers referenced by compatibility layers.
- `options.py` — Accessors bridging to the native options store during the transition.
- `lora.py` — Shared LoRA selection state consumed by legacy surfaces.
- `main.py` — Historical CLI entry helper retained for reference.

## Notes
- New features should target the native services/options stack; keep this package stable until all legacy paths are retired.
- Once the last compatibility consumers migrate, archive this package under `.sangoi/deprecated/` and update the documentation.
