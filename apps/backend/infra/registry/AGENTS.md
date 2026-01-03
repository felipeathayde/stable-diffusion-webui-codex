<!-- tags: backend, infra-registry, assets, text-encoders -->
# apps/backend/infra/registry Overview
Date: 2025-12-05
Owner: Backend Infra Maintainers
Last Review: 2026-01-02
Status: Active

## Purpose
- Host lightweight asset registries for the Codex backend (checkpoints, VAEs, LoRAs, text encoders, tokenizers) that are safe to import without pulling heavy engine/runtime modules.

## Key Files
- `base.py` — Shared `AssetEntry` dataclass and directory iteration helpers.
- `vae.py` — VAE discovery + describe helpers.
- `lora.py` — LoRA discovery + describe helpers (per-engine roots + `apps/paths.json` overrides).
- `embeddings.py` — Textual inversion (TI) discovery + metadata.
- `text_encoders.py` — Vendored text encoder metadata under `apps/backend/huggingface`.
- `text_encoder_dirs.py` — Flat list of vendored text encoder dirs (`org/repo/{text_encoder,t5,clip}`).
- `tokenizers.py` — Vendored tokenizer dirs under `apps/backend/huggingface`.
- `wan22.py` — WAN22-specific registry helpers.

## Notes
- These registries must stay “thin”: no torch/transformers imports, no engine loading. They are intended for inventories, API listings, and future tooling.
- 2025-12-05: Text encoder roots per engine (`*_tenc` in `apps/paths.json`) are now wired into the inventory layer (`apps/backend/inventory/cache.py`); future registry helpers for per-family TE overrides should live here, built on top of `get_paths_for("<engine>_tenc")`.
- 2025-12-29: Added `zimage_tenc` to the text encoder roots registry and `zimage_vae` to VAE discovery so ZImage assets show up in `/api/*` selectors and QuickSettings.
- 2025-12-29: Text encoder root labels (`TextEncoderRoot.name`) now prefer repo-relative paths when roots live under `CODEX_ROOT` (keeps `/api/text-encoders` labels stable and avoids leaking absolute host paths).
- 2026-01-02: Added standardized file header docstrings to `base.py`, `embeddings.py`, `lora.py`, `text_encoder_*.py`, `text_encoders.py`, `tokenizers.py`, `wan22.py`, and package `__init__.py` (doc-only change; part of rollout).
