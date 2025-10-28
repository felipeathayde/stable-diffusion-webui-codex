# apps/backend/huggingface Overview
Date: 2025-10-28
Owner: Runtime Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Stores Codex-managed Hugging Face assets/configuration used for strict/offline execution modes.
- Provides helper functions (`assets.py`) for resolving local mirrors and enforcing our asset policies.

## Notes
- Atualize estes módulos sempre que os requisitos de assets mudarem (ex.: novos mirrors, ajustes de tokenizer/config). Documente alterações relevantes nos task logs.
- Backend loaders esperam a estrutura existente; ao introduzir novos modelos, replique o padrão e mantenha os helpers sincronizados.
