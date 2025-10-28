# apps/backend/runtime/models Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Model registry, loader helpers, and metadata utilities (checkpoints, VAEs, text encoders) shared across engines.

## Notes
- Keep loader logic centralized here (safe loading, dtype inference) and expose only typed interfaces to engines/use cases.
