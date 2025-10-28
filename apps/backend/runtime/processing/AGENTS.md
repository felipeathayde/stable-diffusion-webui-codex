# apps/backend/runtime/processing Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Shared preprocessing utilities (e.g., image conditioning, mask preparation) used before dispatching to engines.

## Notes
- Centralize preprocessing logic here to avoid duplicating conversions in use cases or engines.
