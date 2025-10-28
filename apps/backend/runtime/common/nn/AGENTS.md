# apps/backend/runtime/common/nn Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Shared neural network building blocks (layers, wrappers) used across multiple runtimes/engines.

## Notes
- Keep modules generic; specialize behaviour in the model-specific runtimes instead of here.
