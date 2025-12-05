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
- 2025-12-04: WAN22 inventory now prefers explicit GGUF roots from `apps/paths.json` (`wan22`), falling back to `models/wan22` when not configured, with high/low stage detection preserved.
 - 2025-12-04: Text encoder inventory continues to scan `models/text-encoder` but now also aggregates per-engine roots from `apps/paths.json` (`sd15_tenc`, `sdxl_tenc`, `flux_tenc`, `wan22_tenc`), deduplicating by path so engine-specific TEnc folders show up once under `/api/models/inventory`.
