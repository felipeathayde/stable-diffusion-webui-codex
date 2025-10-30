# apps/backend/runtime/ops Overview
Date: 2025-10-30
Owner: Runtime Maintainers
Last Review: 2025-10-30
Status: Active

## Purpose
- Tensor operations (custom matmul, fused ops, etc.) leveraged by engines and runtimes.

## Notes
- Introduce new ops here and document their expected inputs/outputs to keep usages consistent.
- `operations_bnb.py` now exposes a `BnbQuantConfig` + registry so downstream loaders request 4bit helpers without importing bitsandbytes internals; register additional quant types in the registry (and update documentation) when new variants land.
