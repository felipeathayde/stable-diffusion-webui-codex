<!-- tags: frontend, components, modals, lora, textual-inversion -->
# apps/interface/src/components/modals Overview
Date: 2025-12-04
Owner: Frontend Maintainers
Last Review: 2025-12-04
Status: Active

## Purpose
- Modal components that support prompt editing workflows (checkpoint selection, LoRA picker, textual inversion editor, style editor).

## Notes
- `LoraModal.vue` inserts `<lora:name:weight>` tokens via buttons targeting the positive/negative prompt; views decide which field to update based on the `target` payload.
- Modals should remain dumb about stores; routing of inserted tokens and mutations lives in the view/store layer.

