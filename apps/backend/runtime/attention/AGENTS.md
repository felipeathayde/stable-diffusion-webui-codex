# apps/backend/runtime/attention Overview
Date: 2025-10-28
Owner: Runtime Attention Maintainers
Last Review: 2026-01-24
Status: Active

## Purpose
- Centralizes attention backend selection and related helpers (PyTorch SDPA, optional alt implementations).

## Notes
- Keep new attention kernels registered here so engines/runtime modules can reference a single entrypoint.
- 2026-01-24: Attention dispatch is now runtime-config-driven (no import-time backend binding). xFormers is imported lazily when selected and errors fail loud when unavailable/disabled.
