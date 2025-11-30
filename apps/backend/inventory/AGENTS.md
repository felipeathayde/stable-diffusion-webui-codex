<!-- tags: backend, inventory, models, huggingface, wan22 -->

# apps/backend/inventory Overview
Date: 2025-10-28
Owner: Backend Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Provides lightweight helpers for building and caching backend inventories (used during audits, module parity tracking, and diagnostics).

## Key Files
- `cache.py` — Dataclasses and helpers for persisting inventory snapshots to disk.

## Notes
- When adding new inventory schemas, extend `cache.py` or add adjacent modules here so reporting logic stays centralized.
- 2025-11-30: Inventory Hugging Face root now points at `apps/backend/huggingface` and uses a correct repo-root calculation, keeping metadata listings aligned with WAN22 engines and UI expectations.
