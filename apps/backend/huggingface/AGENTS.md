<!-- tags: backend, huggingface, assets, minimal-files -->

# apps/backend/huggingface Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2026-01-03
Status: Active

## Purpose
- Stores Codex-managed Hugging Face assets/configuration used for strict/offline execution modes.
- Provides helper functions (`assets.py`) for resolving local mirrors and enforcing our asset policies.

## Notes
- Atualize estes módulos sempre que os requisitos de assets mudarem (ex.: novos mirrors, ajustes de tokenizer/config). Documente alterações relevantes nos task logs.
- Backend loaders esperam a estrutura existente; ao introduzir novos modelos, replique o padrão e mantenha os helpers sincronizados.
- 2025-11-30: Call sites import `ensure_repo_minimal_files` directly from `apps.backend.huggingface.assets` to avoid depending on re-exports from the package `__init__`, reducing the chance of `ImportError` when HF helpers drift across environments.
- 2025-12-16: `ensure_repo_minimal_files()` now includes Wan-Animate metadata folders (`image_encoder/`, `image_processor/`) and common processors (`feature_extractor/`) in its allowlist, and treats model-index component configs as required for “config present” checks.
- 2026-01-03: Added standardized file header docstring to `huggingface/__init__.py` (doc-only change; part of rollout).
