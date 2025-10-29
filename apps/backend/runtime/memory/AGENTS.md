# apps/backend/runtime/memory Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2025-10-29
Status: Active

## Purpose
- Memory management policies (VRAM/CPU balance, offload strategies) used by engines during execution.

## Notes
- Keep policy changes centralized here to ensure consistent behaviour across tasks.
- Core dtype/offload helpers (`core_dtype`, `core_initial_load_device`, `core_offload_device`) replace legacy UNet-specific entry points.
