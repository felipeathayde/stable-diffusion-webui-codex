# apps/backend/runtime/processing Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2026-01-02
Status: Active

## Purpose
- Shared preprocessing utilities (e.g., image conditioning, mask preparation) used before dispatching to engines.

## Notes
- Centralize preprocessing logic here to avoid duplicating conversions in use cases or engines.
- `CodexProcessingBase` carries per-job smart flags (`smart_offload`, `smart_fallback`, `smart_cache`) so use-cases and engines can honor request-level overrides without consulting globals directly.
- 2026-01-02: Removed token-merging fields from processing dataclasses (feature is no longer supported).
