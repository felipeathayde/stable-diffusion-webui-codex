<!-- tags: backend, infra-registry, assets, text-encoders -->
# apps/backend/infra/registry Overview
Date: 2025-12-05
Owner: Backend Infra Maintainers
Last Review: 2026-01-20
Status: Active

## Purpose
- Host lightweight asset registries for the Codex backend (checkpoints, VAEs, LoRAs, text encoders, tokenizers) that are safe to import without pulling heavy engine/runtime modules.

## Key Files
- `base.py` — Shared `AssetEntry` dataclass and directory iteration helpers.
- `vae.py` — VAE discovery + describe helpers.
- `lora.py` — LoRA discovery + describe helpers (per-engine roots from `get_paths_for("<engine>_loras")`).
- `embeddings.py` — Textual inversion (TI) discovery + metadata.
- `text_encoder_roots.py` — Engine text encoder roots registry (per-family paths and stable labels).

## Notes
- These registries must stay “thin”: no torch/transformers imports, no engine loading. They are intended for inventories, API listings, and future tooling.
- 2025-12-05: Text encoder roots per engine (`*_tenc` in `apps/paths.json`) are now wired into the inventory layer (`apps/backend/inventory/cache.py`); future registry helpers for per-family TE overrides should live here, built on top of `get_paths_for("<engine>_tenc")`.
- 2026-01-04: Removed direct `apps/paths.json` reads in registries; use `apps/backend/infra/config/paths.py:get_paths_for` instead to keep repo-root expansion/dedup consistent.
- 2026-01-04: LoRA/VAE/WAN22 GGUF discovery now reuses the shared inventory scanners (`apps/backend/inventory/scanners/*`) so `/api/models/inventory` is the single source of truth for roots/extension policy.
- 2026-01-04: Tokenizer discovery now uses the shared vendored HF scanner (`apps/backend/inventory/scanners/vendored_hf.py`) to keep traversal/sorting consistent with inventory metadata.
- 2025-12-29: Added `zimage_tenc` to the text encoder roots registry and `zimage_vae` to VAE discovery so ZImage assets show up in inventory + QuickSettings.
- 2025-12-29: Text encoder root labels (`TextEncoderRoot.name`) now prefer repo-relative paths when roots live under `CODEX_ROOT` (keeps override labels stable and avoids leaking absolute host paths).
- 2026-01-02: Added standardized file header docstrings to `base.py`, `embeddings.py`, `lora.py`, `text_encoder_roots.py`, and package `__init__.py` (doc-only change; part of rollout).
