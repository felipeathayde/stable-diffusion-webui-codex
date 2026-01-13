<!-- tags: frontend, components, modals -->
# apps/interface/src/components/modals Overview
Date: 2025-12-04
Owner: Frontend Maintainers
Last Review: 2026-01-13
Status: Active

## Purpose
- Modal components used across the UI (prompt editing helpers, style editor, and global QuickSettings dialogs).

## Key files
- `apps/interface/src/components/ui/Modal.vue` — shared modal shell (header/body/footer, click-outside backdrop).
- `apps/interface/src/components/modals/CheckpointModal.vue` — checkpoint picker (prompt workflows / views decide insertion targets).
- `apps/interface/src/components/modals/AssetMetadataModal.vue` — read-only JSON metadata viewer for selected checkpoints/assets (QuickSettings info button).
- `apps/interface/src/components/modals/LoraModal.vue` — LoRA picker/insert helpers.
- `apps/interface/src/components/modals/TextualInversionModal.vue` — TI picker/insert helpers.
- `apps/interface/src/components/modals/QuickSettingsOverridesModal.vue` — global device + per-component overrides UI.
- `apps/interface/src/components/modals/StyleEditorModal.vue` — create/edit prompt styles.

## Notes
- Avoid `style="..."` in templates; prefer shared primitives and CSS in `apps/interface/src/styles/**`.
- `LoraModal.vue` inserts `<lora:name:weight>` tokens via buttons targeting the positive/negative prompt; `PromptCard.vue` handles the `target` payload for the main prompt UX (views may still handle it when using the modal directly).
- Keep modals presentational; stores and routing decisions live in views/stores.
- 2026-01-03: Added standardized file header blocks to modal components (doc-only change; part of rollout).
- 2026-01-13: `AssetMetadataModal.vue` adds a nested toggle to switch between raw (flat) and nested (organized) file metadata views.
