# apps/backend/runtime/attention Overview
Date: 2025-10-28
Owner: Runtime Attention Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Centralizes attention backend selection and related helpers (PyTorch SDPA, optional alt implementations).

## Notes
- Keep new attention kernels registered here so engines/runtime modules can reference a single entrypoint.
