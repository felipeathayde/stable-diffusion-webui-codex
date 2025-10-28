# apps/backend/runtime/adapters Overview
Date: 2025-10-28
Owner: Runtime Adapter Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Provides adapter infrastructure (LoRA, SafeTensors, future adapter types) applied to models at runtime.

## Key Files
- `base.py` — Common adapter interfaces.
- `safetensors.py` — SafeTensors-specific helpers used during adapter loading.

## Subdirectories
- `lora/` — Full LoRA pipeline implementation (loader, mapping, ops, type definitions).

## Notes
- Add new adapter families alongside LoRA; keep loader/ops modular so engines can mix and match.
