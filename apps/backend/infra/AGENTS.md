# apps/backend/infra Overview
Date: 2025-10-28
Owner: Backend Infra Maintainers
Last Review: 2026-01-01
Status: Active

## Purpose
- Provides infrastructure utilities shared across backend modules: accelerator detection, configuration parsing, registry tooling.

## Subdirectories
- `accelerators/` — Accelerator discovery and device capability helpers.
- `config/` — Dynamic configuration loaders, CLI argument wiring, and shared config readers (e.g., `paths.json`).
- `registry/` — Shared registry utilities used by engines, services, and runtime modules.

## Notes
- Keep infrastructure logic here to avoid scattering environment/CLI handling across business logic modules.
- When adding new accelerator backends or configuration sources, update these helpers instead of embedding logic in engines.
- 2025-11-03: CLI parser exposes `--debug-conditioning`, mapping to `CODEX_DEBUG_COND` for SDXL conditioning diagnostics.
- 2025-11-03: `--pin-shared-memory` (`CODEX_PIN_SHARED_MEMORY`) controls host pinning for offloaded models; disabled by default on Windows-heavy deployments.
- 2025-12-06: `config/paths.py` agora garante, em best-effort, que roots relativos de modelos definidos em `apps/paths.json` (`sd15_*`, `sdxl_*`, `flux_*`, `wan22_*`) existam sob o repo root, criando diretórios ausentes apenas para entradas relativas; paths absolutos continuam dependendo de provisionamento manual.
- 2025-12-29: Repo root resolution now prefers `CODEX_ROOT` (launchers) over process CWD so configs like `apps/paths.json` and `apps/settings_values.json` stay stable across launch methods.
- 2026-01-01: Added `--gguf-dequantize-upfront` to opt into load-time GGUF dequantization (trades RAM/VRAM for speed; default remains on-the-fly).
