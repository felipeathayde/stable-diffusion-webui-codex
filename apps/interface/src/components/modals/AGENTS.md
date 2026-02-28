<!-- tags: frontend, components, modals -->
# apps/interface/src/components/modals Overview
Date: 2025-12-04
Last Review: 2026-02-21
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
- `LoraModal.vue` emits `<lora:filename:weight>` payloads with `target` for prompt insertion; `PromptCard.vue` and WAN views consume prompt-target semantics consistently.
- Keep modals presentational; stores and routing decisions live in views/stores.
- 2026-01-03: Added standardized file header blocks to modal components (doc-only change; part of rollout).
- 2026-01-13: `AssetMetadataModal.vue` adds in-view controls (Beautify + expand/collapse all) to switch between raw/nested file metadata and manage large trees.
- 2026-02-15: `QuickSettingsOverridesModal.vue` now reflects backend apply metadata; restart warning appears only when `/api/options` reports `restart_required[]`, otherwise it shows hot-apply guidance.
- 2026-02-17: `LoraModal.vue` now supports explicit inventory refresh (`refreshModelInventory`) and surfaces load errors in-modal while emitting filename-based LoRA prompt tokens (`<lora:filename:weight>`); SHA resolution is attached separately at request payload build time.
- 2026-02-28: `LoraModal.vue` continues to emit prompt-target token payloads (`<lora:filename:weight>`) while WAN stage LoRA arrays are derived in `useVideoGeneration` during request assembly.
- 2026-02-21: `StyleEditorModal.vue` now reuses the shared `ui/Modal.vue` shell (teleport + backdrop + footer slot) instead of rendering an ad-hoc modal container.
- 2026-02-21: Shared modal list spacing now uses `.modal-list-section` across checkpoint/TI/LoRA pickers; inline `style="margin-top: ..."` was removed.
