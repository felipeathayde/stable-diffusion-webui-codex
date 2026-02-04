<!-- tags: prompt, components, tiptap -->
# apps/interface/src/components/prompt Overview
Date: 2025-12-03
Last Review: 2026-01-03
Status: Active

## Purpose
- Houses the prompt editor widgets (PromptEditor, PromptBox/Fields) and prompt token parsing/serialization for generation views.

## Notes
- PromptEditor is built on Tiptap StarterKit plus the custom `PromptToken` node.
- Serialization supports both ProseMirror JSON and Node shapes; covered by `PromptToken.test.ts` via Vitest (`npm test`).
- 2025-12-17: `PromptBox.vue` hides the char-count badge when empty (`0 chars`) to reduce visual noise in the WAN tab.
- 2025-12-22: `PromptBox.vue` badge now shows whitespace token count (`tok`) instead of raw character count.
- 2025-12-25: `PromptCard.vue` centralizes the prompt panel header (TI/LoRA/Styles) and negative-prompt visibility; most views default to hidden negative, while SDXL opts in to show it by default.
- 2025-12-26: `PromptCard.vue` panel header now renders the title as plain text (no wrapper `<span>`), consistent with the global panel header convention.
- 2025-12-27: Removed the redundant Checkpoints button from `PromptCard.vue` (checkpoint selection lives in QuickSettings); Styles “New/Apply” are now rendered as input-inline actions.
- 2026-01-03: Added standardized file header blocks to prompt components (non-test files) as part of the rollout (doc-only change).
