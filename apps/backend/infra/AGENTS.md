# apps/backend/infra Overview
Date: 2025-10-28
Owner: Backend Infra Maintainers
Last Review: 2025-11-03
Status: Active

## Purpose
- Provides infrastructure utilities shared across backend modules: accelerator detection, configuration parsing, registry tooling.

## Subdirectories
- `accelerators/` — Accelerator discovery and device capability helpers.
- `config/` — Dynamic configuration loaders and CLI argument wiring.
- `registry/` — Shared registry utilities used by engines, services, and runtime modules.

## Notes
- Keep infrastructure logic here to avoid scattering environment/CLI handling across business logic modules.
- When adding new accelerator backends or configuration sources, update these helpers instead of embedding logic in engines.
- 2025-11-03: CLI parser exposes `--debug-conditioning`, mapping to `CODEX_DEBUG_COND` for SDXL conditioning diagnostics.
- 2025-11-03: `--pin-shared-memory` (`CODEX_PIN_SHARED_MEMORY`) controls host pinning for offloaded models; disabled by default on Windows-heavy deployments.
