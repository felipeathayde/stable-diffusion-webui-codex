# apps/backend/patchers Overview
Date: 2025-10-28
Owner: Backend Runtime Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Hosts runtime patching utilities (LoRA injection, token merging, adapter application) that modify networks or inference behavior after models are loaded.

## Key Files
- `lora_apply.py` — Applies native LoRA selections to loaded networks.
- `token_merging.py` — Implements token merging strategies consumed by engines/use cases.
- Additional patch modules (e.g., adapters) live here as they are ported.

## Notes
- Patchers should operate on runtime objects provided by `runtime/` and `engines/` without duplicating loading logic.
- When introducing new patch behaviour, add explicit configuration flags/options and document them in `.sangoi/backend/`.
