# apps/backend/runtime/memory Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2025-10-31
Status: Active

## Purpose
- Memory management policies (VRAM/CPU balance, offload strategies) used by engines during execution.

## Notes
- `manager.py` hosts `CodexMemoryManager`; `memory_management.py` exposes the compatibility facade used throughout the backend.
- `config.py` defines typed configuration (devices, swap policies, attention backends). Update both docs and `apps/backend/infra/config/args.py` when adding new options.
- Keep policy changes centralized here to ensure consistent behaviour across tasks (engines, patchers, workflows).
- AUTO precision is coordinated via a native ladder: VAE supports bf16→fp16→fp32, diffusion/text encoders bf16→fp16. Fallbacks are handled by `CodexMemoryManager.report_precision_failure` and will refuse to advance when users forced a dtype.
- 2025-11-04: `CodexMemoryManager` now unwraps wrappers (e.g., VAE) to their `ModelPatcher`/`nn.Module` targets before allocation/load so engines can request `load_model_gpu(wrapper)` without tripping AttributeErrors.
