<!-- tags: frontend, components, modals -->
# apps/interface/src/components/modals Overview
Date: 2025-12-04
Owner: Frontend Maintainers
Last Review: 2025-12-23
Status: Active

## Purpose
- Modal components used across the UI (prompt editing helpers, style editor, and global QuickSettings dialogs).

## Key files
- `apps/interface/src/components/ui/Modal.vue` — shared modal shell (header/body/footer, click-outside backdrop).
- `apps/interface/src/components/modals/CheckpointModal.vue` — checkpoint picker (prompt workflows / views decide insertion targets).
- `apps/interface/src/components/modals/LoraModal.vue` — LoRA picker/insert helpers.
- `apps/interface/src/components/modals/TextualInversionModal.vue` — TI picker/insert helpers.
- `apps/interface/src/components/modals/QuickSettingsOverridesModal.vue` — global device + per-component overrides UI.
- `apps/interface/src/components/modals/StyleEditorModal.vue` — create/edit prompt styles.

## Notes
- Avoid `style="..."` in templates; prefer shared primitives and CSS in `apps/interface/src/styles/**`.
- `LoraModal.vue` inserts `<lora:name:weight>` tokens via buttons targeting the positive/negative prompt; views decide which field to update based on the `target` payload.
- Keep modals presentational; stores and routing decisions live in views/stores.
- 2025-12-23: `QuickSettingsWanAssetsModal.vue` now uses the shared `cdx-form-grid` helper for layout (no WAN-specific grid classes).
