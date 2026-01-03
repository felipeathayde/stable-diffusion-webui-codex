<!-- tags: backend, inventory, models, huggingface, wan22 -->

# apps/backend/inventory Overview
Date: 2025-10-28
Owner: Backend Maintainers
Last Review: 2026-01-03
Status: Active

## Purpose
- Provides lightweight helpers for building and caching backend inventories (used during audits, module parity tracking, and diagnostics).

## Key Files
- `cache.py` — Dataclasses and helpers for persisting inventory snapshots to disk.

## Notes
- When adding new inventory schemas, extend `cache.py` or add adjacent modules here so reporting logic stays centralized.
- Reference: `.sangoi/reference/models/model-assets-selection-and-inventory.md` documents the end-to-end contract (inventory → SHA selection → backend resolution).
- 2025-11-30: Inventory Hugging Face root now points at `apps/backend/huggingface` and uses a correct repo-root calculation, keeping metadata listings aligned with WAN22 engines and UI expectations.
- 2025-12-04: WAN22 inventory now prefers explicit GGUF roots from `apps/paths.json` (`wan22`), falling back to `models/wan22` when not configured, with high/low stage detection preserved.
- 2025-12-04: Text encoder inventory continues to scan `models/text-encoder` but now also aggregates per-engine roots from `apps/paths.json` (`sd15_tenc`, `sdxl_tenc`, `flux_tenc`, `wan22_tenc`, `zimage_tenc`), deduplicating by path so engine-specific TEnc folders show up once under `/api/models/inventory`.
- 2025-12-04: Inventory also appends engine-specific VAEs from `apps/paths.json` (`flux_vae`, `zimage_vae`) so Flux/ZImage dropdowns can list concrete files without filename heuristics.
- 2025-12-06: Model inventory is now pre-warmed during backend bootstrap (`_bootstrap_runtime` calls `inventory.cache.refresh()`), so the first `/api/models/inventory` request no longer pays the full filesystem scan cost on demand.
- 2025-12-29: Inventory repo root now prefers `CODEX_ROOT` so scans don’t depend on the backend process CWD.
- 2026-01-03: Added standardized file header docstring to `cache.py` (doc-only change; part of rollout).
