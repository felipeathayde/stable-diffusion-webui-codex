# gitignore Policy — Stable Diffusion WebUI Codex
Date: 2025-12-04
Owner: Repository Maintainers
Last Review: 2025-12-04
Status: Active

## Purpose
- Document the intent behind `.gitignore` so large artefacts and local caches stay out of the repository while keeping all source and documentation tracked.

## Scope
- `.gitignore` governs which files are ignored by Git.
- This document explains *why* certain patterns are ignored and how to extend the policy safely.

## Principles
- **No models in Git:** model weights, checkpoints, and large binary artefacts (e.g., `models/`, `*.ckpt`, `*.safetensors`) must remain untracked.
- **No caches or build outputs:** bytecode caches (`__pycache__/`), node modules (`node_modules/`), frontend bundles (`apps/interface/dist/`), and temporary directories (`tmp/`, `.pytest_cache/`) stay ignored.
- **Docs and configs are tracked:** Markdown docs under `.sangoi/**`, configuration files (`*.json`, `*.toml`, `*.yaml`), and source code are always included.
- **Tests are tracked:** repository test sources under `tests/` are kept in version control; only test caches/outputs are ignored.

## Extending the ignore set
- Prefer directory-level ignores (e.g., `apps/interface/dist/`) over broad `*` patterns.
- When adding a new ignore rule:
  - Confirm the files are reproducible or environment-local.
  - Update this file with a short note if the pattern is non-obvious.
  - Avoid ignoring generic extensions that might hide source (e.g., `*.py`, `*.ts`).

## Related Notes
- Large artefacts, outputs, caches, and heavy model directories stay untracked per this policy, as referenced in `AGENTS.md`.
- If you are unsure whether something should be ignored, log the question in `.sangoi/task-logs/` and coordinate with maintainers before changing `.gitignore`.
