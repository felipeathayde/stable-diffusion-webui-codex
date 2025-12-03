<!-- tags: prompt, components, tiptap -->
# apps/interface/src/components/prompt Overview
Date: 2025-12-03
Owner: Frontend Maintainers
Last Review: 2025-12-03
Status: Active

## Purpose
- Houses the prompt editor widgets (PromptEditor, PromptBox/Fields) and prompt token parsing/serialization for generation views.

## Notes
- PromptEditor is built on Tiptap StarterKit plus the custom `PromptToken` node.
- Serialization supports both ProseMirror JSON and Node shapes; covered by `PromptToken.test.ts` via Vitest (`npm test`).
