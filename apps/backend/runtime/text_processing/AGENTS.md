# apps/backend/runtime/text_processing Overview
Date: 2025-10-28
Owner: Runtime Text Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Text processing utilities: prompt parsing, emphasis, textual inversion, CLIP/T5 helpers used across engines and adapters.

## Notes
- Keep token/embedding handling centralized here to avoid drift between engines.
- CLIP classic engines now consult the AUTO precision ladder; encoding retries will fall back bf16→fp16 with loud logging, while manual precision flags bypass the ladder.
