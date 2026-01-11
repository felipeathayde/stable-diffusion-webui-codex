<!-- tags: backend, inventory, scanners, assets -->
# apps/backend/inventory/scanners Overview
Date: 2026-01-04
Owner: Backend Maintainers
Last Review: 2026-01-04
Status: Active

## Purpose
- Shared, import-light filesystem scanners for model assets (VAEs, LoRAs, etc.).
- Provides a single place to define root/extension policies so `inventory/cache.py` and UI-facing registries don’t drift.

## Key files
- `base.py`: Shared helpers for resolving repo/models roots and walking directories.
- `vaes.py`: VAE file discovery policy (per-family `*_vae` roots from `apps/paths.json`).
- `loras.py`: LoRA file discovery policy (per-family `*_loras` roots from `apps/paths.json`).
- `text_encoders.py`: Text encoder weight discovery policy (per-family `*_tenc` roots from `apps/paths.json`).
- `wan22_gguf.py`: WAN22 stage GGUF discovery policy (paths.json `wan22_ckpt`) + stage classifier.
- `vendored_hf.py`: Shared `{org}/{repo}` directory walk for vendored HF roots (tokenizers + metadata inventory).

## Notes
- Scanners must stay lightweight (no torch/transformers imports). Hashing and heavy metadata extraction lives in inventory/cache or runtime.
- Root configuration comes from `apps/backend/infra/config/paths.py:get_paths_for` and `CODEX_ROOT` via `infra/config/repo_root.py`.
- Scanners intentionally ignore ad-hoc files under `models/` (only explicit roots from `apps/paths.json` are scanned).
